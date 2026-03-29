cmake_minimum_required(VERSION 3.18)
project(NukeFlareSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# All compilation units (including CUDA) must emit position-independent code,
# which is required for shared libraries on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Nuke version — override on the cmake command line:
#   cmake -DNUKE_VERSION=14.1v8 ..
#   cmake -DNUKE_VERSION=16.0v1 ..
#   cmake -DNUKE_VERSION=17.0v1 ..
# ---------------------------------------------------------------------------
if(NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION "14.1v8")
endif()

# ---------------------------------------------------------------------------
# CUDA architectures
#
# Minimum: sm_70 (Volta) — required for __match_any_sync warp intrinsics
# used in the ghost scatter kernel.
#
# sm_70  — Volta         (V100, Titan V)                CUDA 9.0+
# sm_75  — Turing        (RTX 2000, T4)                 CUDA 10.0+
# sm_86  — Ampere        (RTX 3000, A4500/A5000/A6000)  CUDA 11.1+
# sm_89  — Ada Lovelace  (RTX 4000)                     CUDA 11.8+
# sm_90  — Hopper        (H100)                         CUDA 12.0+
# sm_100 — Blackwell     (B100/B200, RTX 5000)          CUDA 12.8+
# sm_120 — Blackwell     (GB10, Jetson Thor)             CUDA 12.8+
#
# Default: all architectures.  Requires CUDA 12.8+ (set via PATH).
# For older toolkits pass only the supported subset:
#   CUDA 12.1: cmake -DCMAKE_CUDA_ARCHITECTURES="86;89;90" ..
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 90 100 120)
endif()

# Reject any architecture below sm_70 — warp-shuffle intrinsics won't compile.
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_arch LESS 70)
        message(FATAL_ERROR
            "FlareSim requires sm_70+ (Volta or newer).  "
            "Architecture sm_${_arch} is not supported.\n"
            "  Remove it from CMAKE_CUDA_ARCHITECTURES or upgrade your target GPU.")
    endif()
endforeach()

# Static CUDA runtime — bakes cudart into the plugin binary so users do not
# need a CUDA runtime installed.  On Windows we keep Shared to avoid any
# potential CRT conflict with Nuke's /MD runtime.
if(WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)   # use cudart.dll; avoids LIBCMT conflict with Nuke's /MD
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)   # bake cudart in; no runtime dependency on Linux
endif()

# Match Nuke's MSVC runtime — must be MultiThreaded DLL (/MD).
# Mixing runtimes causes heap corruption and hard-to-diagnose crashes.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ---------------------------------------------------------------------------
# Paths — override on the cmake command line if Nuke is installed elsewhere
#
# Rocky Linux / CentOS typical install:  /usr/local/Nuke14.1v8
# ---------------------------------------------------------------------------
if(WIN32)
    set(NDK_ROOT "C:/Program Files/Nuke${NUKE_VERSION}/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "C:/Program Files/Nuke${NUKE_VERSION}"
        CACHE PATH "Nuke library directory (contains DDImage.lib / Ndk.lib)")
else()
    set(NDK_ROOT "/usr/local/Nuke${NUKE_VERSION}/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "/usr/local/Nuke${NUKE_VERSION}"
        CACHE PATH "Nuke library directory (contains libDDImage.so)")
endif()

# Sanity check — catch typos and missing installs early.
if(NOT EXISTS "${NDK_ROOT}")
    message(WARNING "Nuke NDK include directory not found: ${NDK_ROOT}\n"
                    "  Set -DNUKE_VERSION=<ver> or -DNDK_ROOT=<path>")
endif()
if(NOT WIN32 AND NOT EXISTS "${NUKE_LIB_DIR}/libDDImage.so")
    message(FATAL_ERROR
        "Nuke not found at ${NUKE_LIB_DIR} (no libDDImage.so).\n"
        "  Set -DNUKE_VERSION=<ver> or -DNUKE_LIB_DIR=<path>\n"
        "  Example: cmake -DNUKE_VERSION=14.1v8 ..")
endif()

# ---------------------------------------------------------------------------
# Detect CUDA toolkit version for the build summary
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --version
    OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _match "${NVCC_VERSION_OUTPUT}")
set(CUDA_TOOLKIT_VERSION "${CMAKE_MATCH_1}")

message(STATUS "")
message(STATUS "========================================")
message(STATUS "FlareSim Build Configuration:")
message(STATUS "  Nuke version : ${NUKE_VERSION}")
message(STATUS "  NDK include  : ${NDK_ROOT}")
message(STATUS "  Nuke libs    : ${NUKE_LIB_DIR}")
message(STATUS "  CUDA toolkit : ${CUDA_TOOLKIT_VERSION}")
message(STATUS "  CUDA archs   : ${CMAKE_CUDA_ARCHITECTURES}")
if(WIN32)
    message(STATUS "  CUDA runtime : Shared (Windows)")
else()
    message(STATUS "  CUDA runtime : Static (baked into plugin)")
endif()
message(STATUS "========================================")
message(STATUS "")

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
add_library(NukeFlareSim SHARED
    src/FlareSim.cpp
    src/ghost_cuda.cu
    src/blur_cuda.cu
    src/lens.cpp
    src/trace.cpp
    src/ghost.cpp
)

target_include_directories(NukeFlareSim PRIVATE
    src
    "${NDK_ROOT}"
)

if(WIN32)
    target_compile_definitions(NukeFlareSim PRIVATE
        NOMINMAX            # required by Nuke NDK (fnVC.h enforces this)
        _USE_MATH_DEFINES   # required by Nuke NDK (fnVC.h enforces this); exposes M_PI etc.
        WIN32_LEAN_AND_MEAN
    )
endif()

# Nuke 14.x and earlier were built with the old pre-C++11 GCC ABI.
# Nuke 15+ uses the new ABI.  The flag must apply to ALL languages
# (CXX *and* CUDA) — otherwise the CUDA .cu files compile launch_ghost_cuda()
# with a different std::string mangling than the CXX .cpp files that call it,
# producing an "undefined symbol" error at load time.
#
# Extract the major version number from NUKE_VERSION (e.g. "14.1v8" → "14").
string(REGEX MATCH "^[0-9]+" NUKE_MAJOR "${NUKE_VERSION}")
if(NUKE_MAJOR AND NUKE_MAJOR LESS 15)
    message(STATUS "  Nuke ${NUKE_MAJOR}.x detected — using old GCC ABI (_GLIBCXX_USE_CXX11_ABI=0)")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# CXX warning flags — platform-conditional.
# MSVC: treat warnings as informational only; keep /W3.
#   /wd4996 suppresses STL3DAllocator deprecation warnings that originate
#   inside Nuke's own DDImage headers (Attribute.h, Primitive.h, Allocators.h).
# GCC/Clang: standard warnings.
if(MSVC)
    target_compile_options(NukeFlareSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4996>
    )
else()
    target_compile_options(NukeFlareSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-deprecated-declarations>
    )
endif()

# CUDA: allow constexpr functions to be used on device without __device__ tags;
# required for some C++17 standard-library helpers used in device-side headers.
target_compile_options(NukeFlareSim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(WIN32)
    target_link_libraries(NukeFlareSim PRIVATE
        "${NUKE_LIB_DIR}/DDImage.lib"
        "${NUKE_LIB_DIR}/Ndk.lib"
    )
else()
    target_link_libraries(NukeFlareSim PRIVATE
        "${NUKE_LIB_DIR}/libDDImage.so"
    )
    # Not all Nuke versions ship a separate libNdk.so on Linux.
    if(EXISTS "${NUKE_LIB_DIR}/libNdk.so")
        target_link_libraries(NukeFlareSim PRIVATE
            "${NUKE_LIB_DIR}/libNdk.so"
        )
    endif()
endif()

# Nuke expects the plugin as a plain shared library with no "lib" prefix.
# CMAKE_SHARED_LIBRARY_SUFFIX gives ".dll" on Windows and ".so" on Linux.
set_target_properties(NukeFlareSim PROPERTIES
    OUTPUT_NAME "FlareSim"     # plugin name must match the registered node class name
    PREFIX ""
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
    CUDA_SEPARABLE_COMPILATION OFF
)

# ---------------------------------------------------------------------------
# Install helper — copies the plugin into the directory pointed to by
# the NUKE_PLUGIN_DIR cache variable if provided.
# Usage: cmake -DNUKE_PLUGIN_DIR="$HOME/.nuke/plugins" ..
# ---------------------------------------------------------------------------
if(DEFINED NUKE_PLUGIN_DIR)
    install(TARGETS NukeFlareSim DESTINATION "${NUKE_PLUGIN_DIR}")
endif()
