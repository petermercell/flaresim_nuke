cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# CRITICAL: Set MSVC runtime library policy BEFORE project()
# This enables modern runtime library control via CMAKE_MSVC_RUNTIME_LIBRARY.
# ============================================================================
cmake_policy(SET CMP0091 NEW)

project(NukeFlareSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# All compilation units (including CUDA) must emit position-independent code,
# which is required for shared libraries on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Nuke version — override on the cmake command line:
#   cmake -DNUKE_VERSION=14.1v8 ..
#   cmake -DNUKE_VERSION=16.0v6 ..
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
# Default: safe subset for CUDA 12.2.
# For newer toolkits add sm_100 and sm_120.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 90)
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

# ============================================================================
# Platform-specific runtime and compiler configuration
# ============================================================================
# Static CUDA runtime — bakes cudart into the plugin binary on ALL platforms.
# No cudart64_*.dll dependency at runtime on Windows, no libcudart.so on Linux.
# CMP0091 NEW + CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" ensures the
# static cudart links with /MD (matching Nuke), avoiding /MT vs /MD conflicts.
# ============================================================================
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

if(WIN32)
    # ---- Windows / MSVC ----

    # Match Nuke's MSVC runtime — MultiThreaded DLL (/MD).
    # Mixing runtimes (/MT vs /MD) causes heap corruption.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    # Export all symbols from shared libraries — needed because our .cu files
    # define functions (ensure_pinned_output_fp16 etc.) that .cpp files call.
    # On Linux this is handled by __attribute__((visibility("default"))).
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

    # Global definitions for Windows builds (safe for all languages including CUDA)
    add_definitions(
        -DNOMINMAX
        -D_USE_MATH_DEFINES
        -DWIN32_LEAN_AND_MEAN
        -D_CRT_SECURE_NO_WARNINGS
    )

    # NOTE: MSVC /W3 /O2 etc are set per-target below using
    # $<$<COMPILE_LANGUAGE:CXX>:...> to avoid passing them to nvcc.
    # nvcc gets its own flags via -Xcompiler in the per-target loop.

endif()

# ---------------------------------------------------------------------------
# Paths — override on the cmake command line if Nuke is installed elsewhere
#
# Linux typical:   /usr/local/Nuke14.1v8  or  /opt/Nuke17.0v1
# Windows typical: C:/Program Files/Nuke17.0v1
# ---------------------------------------------------------------------------
if(WIN32)
    set(NDK_ROOT "C:/Program Files/Nuke${NUKE_VERSION}/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "C:/Program Files/Nuke${NUKE_VERSION}"
        CACHE PATH "Nuke library directory (contains DDImage.lib)")
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

if(WIN32)
    if(NOT EXISTS "${NUKE_LIB_DIR}/DDImage.lib")
        message(FATAL_ERROR
            "Nuke not found at ${NUKE_LIB_DIR} (no DDImage.lib).\n"
            "  Set -DNUKE_VERSION=<ver> or -DNUKE_LIB_DIR=<path>\n"
            "  Example: cmake -DNUKE_VERSION=16.0v6 ..\n"
            "  Or:      cmake -DNUKE_LIB_DIR=\"D:/Nuke/17.0v1\" ..")
    endif()
else()
    if(NOT EXISTS "${NUKE_LIB_DIR}/libDDImage.so")
        message(FATAL_ERROR
            "Nuke not found at ${NUKE_LIB_DIR} (no libDDImage.so).\n"
            "  Set -DNUKE_VERSION=<ver> or -DNUKE_LIB_DIR=<path>\n"
            "  Example: cmake -DNUKE_VERSION=14.1v8 ..")
    endif()
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

# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "FlareSim Build Configuration:")
message(STATUS "  Nodes        : FlareSim, FlareSim3D")
if(WIN32)
    message(STATUS "  Outputs      : FlareSim.dll, FlareSim3D.dll")
else()
    message(STATUS "  Outputs      : FlareSim.so, FlareSim3D.so")
endif()
message(STATUS "  Nuke version : ${NUKE_VERSION}")
message(STATUS "  NDK include  : ${NDK_ROOT}")
message(STATUS "  Nuke libs    : ${NUKE_LIB_DIR}")
message(STATUS "  CUDA toolkit : ${CUDA_TOOLKIT_VERSION}")
message(STATUS "  CUDA archs   : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA runtime : Static (baked into plugin)")
if(WIN32)
    message(STATUS "  MSVC runtime : MultiThreadedDLL (/MD)")
    if(MSVC)
        message(STATUS "  Compiler     : MSVC ${MSVC_VERSION}")
    endif()
endif()
message(STATUS "========================================")
message(STATUS "")

# ---------------------------------------------------------------------------
# Common source files (shared between FlareSim and FlareSim3D)
# ---------------------------------------------------------------------------
set(COMMON_SOURCES
    src/ghost_cuda.cu
    src/blur_cuda.cu
    src/lens.cpp
    src/trace.cpp
    src/ghost.cpp
)

# ---------------------------------------------------------------------------
# FlareSim — 2D source (manual XY knob)
# ---------------------------------------------------------------------------
add_library(FlareSim SHARED
    src/FlareSim.cpp
    ${COMMON_SOURCES}
)

# ---------------------------------------------------------------------------
# FlareSim3D — 3D source (Camera + Axis inputs)
# ---------------------------------------------------------------------------
add_library(FlareSim3D SHARED
    src/FlareSim3D.cpp
    ${COMMON_SOURCES}
)

# ---------------------------------------------------------------------------
# Shared settings for both targets
# ---------------------------------------------------------------------------
foreach(_tgt FlareSim FlareSim3D)

    target_include_directories(${_tgt} PRIVATE
        src
        "${NDK_ROOT}"
    )

    # CXX warning and optimisation flags
    if(MSVC)
        target_compile_options(${_tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /Ob2 /EHsc /FS /nologo /MP>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4996 /wd4244 /wd4267 /wd4305 /wd4251>
        )
    else()
        target_compile_options(${_tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-deprecated-declarations>
        )
    endif()

    # CUDA: allow constexpr on device without __device__ tags
    target_compile_options(${_tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # CUDA + MSVC: forward compiler flags to the host compiler via -Xcompiler
    if(MSVC)
        target_compile_options(${_tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -Xcompiler=/W3
                -Xcompiler=/wd4996
                -Xcompiler=/wd4244
                -Xcompiler=/wd4267
                -Xcompiler=/O2
                -Xcompiler=/EHsc
                -Xcompiler=/FS
            >
        )
    endif()

    # ---- Linking ----
    if(WIN32)
        target_link_libraries(${_tgt} PRIVATE
            "${NUKE_LIB_DIR}/DDImage.lib"
        )
        if(EXISTS "${NUKE_LIB_DIR}/Ndk.lib")
            target_link_libraries(${_tgt} PRIVATE
                "${NUKE_LIB_DIR}/Ndk.lib"
            )
        endif()
    else()
        target_link_libraries(${_tgt} PRIVATE
            "${NUKE_LIB_DIR}/libDDImage.so"
        )
        if(EXISTS "${NUKE_LIB_DIR}/libNdk.so")
            target_link_libraries(${_tgt} PRIVATE
                "${NUKE_LIB_DIR}/libNdk.so"
            )
        endif()
    endif()

    set_target_properties(${_tgt} PROPERTIES
        PREFIX ""
        SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
        CUDA_SEPARABLE_COMPILATION OFF
    )

endforeach()

# Per-target output names (must match Nuke node class names)
set_target_properties(FlareSim   PROPERTIES OUTPUT_NAME "FlareSim")
set_target_properties(FlareSim3D PROPERTIES OUTPUT_NAME "FlareSim3D")

# Nuke 14.x and earlier were built with the old pre-C++11 GCC ABI.
# Must apply to ALL languages (CXX *and* CUDA).  Has no effect on MSVC.
string(REGEX MATCH "^[0-9]+" NUKE_MAJOR "${NUKE_VERSION}")
if(NUKE_MAJOR AND NUKE_MAJOR LESS 15)
    message(STATUS "  Nuke ${NUKE_MAJOR}.x detected — using old GCC ABI (_GLIBCXX_USE_CXX11_ABI=0)")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
if(WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/.nuke"
            CACHE PATH "Install path" FORCE)
    endif()
    install(TARGETS FlareSim FlareSim3D
        RUNTIME DESTINATION plugins
        LIBRARY DESTINATION plugins
    )
elseif(DEFINED NUKE_PLUGIN_DIR)
    install(TARGETS FlareSim FlareSim3D DESTINATION "${NUKE_PLUGIN_DIR}")
endif()