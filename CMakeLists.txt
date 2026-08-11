cmake_minimum_required(VERSION 3.16)
project(aoclsparse_portable VERSION 5.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SUPPORT_OMP "Enable OpenMP support" ON)

# ARM-compatible compile flags (no x86-specific flags)
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -ffp-contract=fast")
endif()

# ARM NEON SIMD acceleration
# On AArch64, NEON is mandatory (__ARM_NEON defined by default).
# On ARMv7-A, NEON requires explicit -mfpu=neon.
# We use -mcpu=native for best instruction selection on the build host.
option(ENABLE_ARM_SIMD "Enable ARM NEON/SVE SIMD acceleration" ON)
if(ENABLE_ARM_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armv7l|armv7-a")
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -mfpu=neon -mfloat-abi=hard")
        endif()
    endif()
endif()

# Optional SVE (Scalable Vector Extension) auto-vectorization — AArch64 only
option(ENABLE_SVE "Enable SVE auto-vectorization (AArch64 only)" OFF)
if(ENABLE_SVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve")
    endif()
endif()

if(SUPPORT_OMP)
    find_package(OpenMP REQUIRED)
endif()

# Version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/aoclsparse_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/aoclsparse_version.h
)

# Shared library
add_library(aoclsparse_portable SHARED
    src/auxiliary/aoclsparse_auxiliary.cpp
    src/analysis/aoclsparse_csr_util.cpp
    src/level2/aoclsparse_mv.cpp
    src/level3/aoclsparse_csradd.cpp
    src/level3/aoclsparse_sp2m.cpp
    src/level3/aoclsparse_csr2m.cpp
    src/level3/aoclsparse_sp2md.cpp
    src/level3/aoclsparse_csrmm.cpp
)

target_include_directories(aoclsparse_portable
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/auxiliary
        ${CMAKE_CURRENT_SOURCE_DIR}/src/analysis
        ${CMAKE_CURRENT_SOURCE_DIR}/src/conversion
        ${CMAKE_CURRENT_SOURCE_DIR}/src/level2
        ${CMAKE_CURRENT_SOURCE_DIR}/src/level3
)

if(SUPPORT_OMP)
    target_link_libraries(aoclsparse_portable PRIVATE OpenMP::OpenMP_CXX)
endif()
