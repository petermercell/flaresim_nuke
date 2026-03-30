cmake_minimum_required(VERSION 3.18)

# macOS deployment target + architecture MUST be set before project()
# so CMake configures the compiler correctly from the start.
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build for Apple Silicon")

project(NukeFlareSim LANGUAGES CXX OBJCXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# ---------------------------------------------------------------------------
# Nuke version — override on the cmake command line:
#   cmake -DNUKE_VERSION=15.1v10 ..
#   cmake -DNUKE_VERSION=17.0v1 ..
# ---------------------------------------------------------------------------
if(NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION "17.0v1")
endif()

# ---------------------------------------------------------------------------
# Nuke NDK paths (macOS .app bundle layout)
# ---------------------------------------------------------------------------
if(NOT DEFINED NUKE_ROOT)
    # Try common macOS Nuke locations
    foreach(_ver ${NUKE_VERSION} "17.0v1" "16.0v1" "15.1v10")
        if(EXISTS "/Applications/Nuke${_ver}/Nuke${_ver}.app/Contents/MacOS")
            set(NUKE_ROOT "/Applications/Nuke${_ver}/Nuke${_ver}.app/Contents/MacOS")
            break()
        endif()
    endforeach()
    if(NOT DEFINED NUKE_ROOT)
        message(FATAL_ERROR "NUKE_ROOT not set and Nuke not found.\n"
                "Set it with: -DNUKE_ROOT=/Applications/NukeXX.Xv1/NukeXX.Xv1.app/Contents/MacOS")
    endif()
endif()

message(STATUS "Nuke root: ${NUKE_ROOT}")

if(NOT EXISTS "${NUKE_ROOT}/include/DDImage/Iop.h")
    message(FATAL_ERROR "Invalid NUKE_ROOT: ${NUKE_ROOT} - DDImage headers not found")
endif()

set(NUKE_INCLUDE_DIRS "${NUKE_ROOT}/include")

find_library(DDIMAGE_LIBRARY
    NAMES DDImage
    PATHS "${NUKE_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED
)

# ---------------------------------------------------------------------------
# macOS Frameworks
# ---------------------------------------------------------------------------
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

message(STATUS "Metal:      ${METAL_FRAMEWORK}")
message(STATUS "Foundation: ${FOUNDATION_FRAMEWORK}")

# ---------------------------------------------------------------------------
# Compile Metal shaders → .metallib and embed via assembly
#
# Pipeline:
#   1. xcrun metal -c  →  .air  (intermediate representation)
#   2. xcrun metallib  →  .metallib  (compiled library)
#   3. .incbin         →  embedded in the plugin binary
# ---------------------------------------------------------------------------

set(METAL_SOURCE "${CMAKE_SOURCE_DIR}/src/flare_shaders.metal")
set(METAL_AIR    "${CMAKE_BINARY_DIR}/flare_shaders.air")
set(METALLIB     "${CMAKE_BINARY_DIR}/flare_shaders.metallib")
set(EMBED_ASM    "${CMAKE_BINARY_DIR}/embed_metallib.s")

# Step 1: Compile .metal → .air
add_custom_command(
    OUTPUT "${METAL_AIR}"
    COMMAND xcrun metal
        -std=metal3.0
        -target air64-apple-macos13.0
        -O2
        -c "${METAL_SOURCE}"
        -o "${METAL_AIR}"
    DEPENDS "${METAL_SOURCE}"
    COMMENT "Compiling Metal shaders → .air"
    VERBATIM
)

# Step 2: Link .air → .metallib
add_custom_command(
    OUTPUT "${METALLIB}"
    COMMAND xcrun metallib
        "${METAL_AIR}"
        -o "${METALLIB}"
    DEPENDS "${METAL_AIR}"
    COMMENT "Linking Metal library → .metallib"
    VERBATIM
)

# Step 3: Generate assembly to embed the metallib via .incbin
file(WRITE "${EMBED_ASM}"
    ".section __DATA,__const\n"
    ".globl __binary_flare_shaders_metallib_start\n"
    ".globl __binary_flare_shaders_metallib_end\n"
    ".align 4\n"
    "__binary_flare_shaders_metallib_start:\n"
    ".incbin \"${METALLIB}\"\n"
    "__binary_flare_shaders_metallib_end:\n"
)

set_source_files_properties("${EMBED_ASM}" PROPERTIES
    OBJECT_DEPENDS "${METALLIB}"
    LANGUAGE ASM
)

# ---------------------------------------------------------------------------
# Common source files (shared between FlareSim and FlareSim3D)
# ---------------------------------------------------------------------------

# Ensure metallib is built before either plugin
add_custom_target(embedded_metallib DEPENDS "${METALLIB}")

set(COMMON_SOURCES
    src/ghost_metal.mm
    src/lens.cpp
    src/trace.cpp
    src/ghost.cpp
    "${EMBED_ASM}"
)

# ---------------------------------------------------------------------------
# Build plugins — one .dylib per node so Nuke discovers both by filename
# ---------------------------------------------------------------------------

# ---- FlareSim ----
add_library(FlareSim SHARED src/FlareSim.mm ${COMMON_SOURCES})

add_dependencies(FlareSim embedded_metallib)

target_include_directories(FlareSim PRIVATE
    src
    "${NUKE_INCLUDE_DIRS}"
)

target_link_libraries(FlareSim PRIVATE
    ${DDIMAGE_LIBRARY}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(FlareSim PROPERTIES
    OUTPUT_NAME "FlareSim"
    PREFIX ""
    SUFFIX ".dylib"
    MACOSX_RPATH TRUE
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ---- FlareSim3D ----
add_library(FlareSim3D SHARED src/FlareSim3D.mm ${COMMON_SOURCES})

add_dependencies(FlareSim3D embedded_metallib)

target_include_directories(FlareSim3D PRIVATE
    src
    "${NUKE_INCLUDE_DIRS}"
)

target_link_libraries(FlareSim3D PRIVATE
    ${DDIMAGE_LIBRARY}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(FlareSim3D PROPERTIES
    OUTPUT_NAME "FlareSim3D"
    PREFIX ""
    SUFFIX ".dylib"
    MACOSX_RPATH TRUE
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ---- Shared compile options ----
foreach(_tgt FlareSim FlareSim3D)
    target_compile_options(${_tgt} PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-deprecated-declarations
        -fvisibility=hidden
        -O3
    )
endforeach()

# Objective-C++ flags (no ARC — manual memory management to match Nuke NDK)
set_source_files_properties(
    src/FlareSim.mm src/FlareSim3D.mm src/ghost_metal.mm
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
if(DEFINED NUKE_PLUGIN_DIR)
    install(TARGETS FlareSim FlareSim3D DESTINATION "${NUKE_PLUGIN_DIR}")
endif()

install(TARGETS FlareSim FlareSim3D LIBRARY DESTINATION .)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "FlareSim Metal — Build Configuration")
message(STATUS "========================================")
message(STATUS "  Nodes        : FlareSim, FlareSim3D")
message(STATUS "  Outputs      : FlareSim.dylib, FlareSim3D.dylib")
message(STATUS "  Nuke version : ${NUKE_VERSION}")
message(STATUS "  Nuke root    : ${NUKE_ROOT}")
message(STATUS "  macOS target : ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "  Architecture : ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "  Metal shaders: ${METAL_SOURCE}")
message(STATUS "  Compute      : Apple Silicon (Metal 3.0)")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
message(STATUS "")