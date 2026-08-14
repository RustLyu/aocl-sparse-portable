#!/bin/bash
# Build script for AOCL-Sparse Portable (ARMv7-A cross-compilation)
# Produces shared library + test_aocl_sparse executable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_arm_gcc"
TEST_DIR="${SCRIPT_DIR}/test_spmm"
TEST_BUILD_DIR="${TEST_DIR}/build"

# ------------------------------------------------------------------
# Toolchain configuration
# ------------------------------------------------------------------
TOOLCHAIN_DIR="${HOME}/tools/petaLinux/tools/linux-i386/gcc-arm-linux-gnueabi"
TOOLCHAIN_PREFIX="${TOOLCHAIN_DIR}/bin/arm-linux-gnueabihf"
TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchain-arm-gnueabihf.cmake"

# --- CPU / FPU flags (adjust to match your target) -----------------
CPU_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=hard"

# --- OpenMP -------------------------------------------------------
OMP_LIB="${TOOLCHAIN_DIR}/arm-linux-gnueabihf/lib/libgomp.so"

# --- Install prefix (change to suit your toolchain / sysroot) -----
INSTALL_PREFIX="${HOME}/opt/arm-v7-a/aocl-sparse-portable"

# ------------------------------------------------------------------
# Step 1: Cross-compile the shared library
# ------------------------------------------------------------------
echo "=============================================="
echo "Step 1: Building libaoclsparse_portable.so ..."
echo "=============================================="

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_CXX_FLAGS="${CPU_FLAGS}" \
    -DOpenMP_CXX_FLAGS="-fopenmp" \
    -DOpenMP_CXX_LIB_NAMES="gomp" \
    -DOpenMP_gomp_LIBRARY="${OMP_LIB}" \
    -DENABLE_ARM_SIMD=OFF \
    -DSUPPORT_OMP=ON

make -C "${BUILD_DIR}" -j$(nproc)

echo ""
echo "Library built: ${BUILD_DIR}/libaoclsparse_portable.so"
echo ""

# ------------------------------------------------------------------
# Step 2: Cross-compile the test_aocl_sparse executable
# ------------------------------------------------------------------
echo "=============================================="
echo "Step 2: Building test_aocl_sparse ..."
echo "=============================================="

mkdir -p "${TEST_BUILD_DIR}"
cmake -S "${TEST_DIR}" -B "${TEST_BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_C_FLAGS="${CPU_FLAGS}"

make -C "${TEST_BUILD_DIR}" -j$(nproc)

echo ""
echo "Test built: ${TEST_BUILD_DIR}/test_aocl_sparse"
echo ""

# ------------------------------------------------------------------
# Step 3: Install to prefix
# ------------------------------------------------------------------
echo "=============================================="
echo "Step 3: Installing to ${INSTALL_PREFIX} ..."
echo "=============================================="

mkdir -p "${INSTALL_PREFIX}/lib" "${INSTALL_PREFIX}/include" "${INSTALL_PREFIX}/lib/cmake/aoclsparse_portable"
cp "${BUILD_DIR}/libaoclsparse_portable.so" "${INSTALL_PREFIX}/lib/"
cp "${SCRIPT_DIR}/include/"*.h   "${INSTALL_PREFIX}/include/"
cp "${SCRIPT_DIR}/include/"*.hpp "${INSTALL_PREFIX}/include/"
cp "${BUILD_DIR}/include/aoclsparse_version.h" "${INSTALL_PREFIX}/include/"
cp "${SCRIPT_DIR}/cmake/aoclsparse_portable-config.cmake.in" \
   "${INSTALL_PREFIX}/lib/cmake/aoclsparse_portable/aoclsparse_portable-config.cmake"

echo ""
echo "Installed files:"
find "${INSTALL_PREFIX}" -type f | sort
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "=============================================="
echo "Build complete"
echo "=============================================="
file "${BUILD_DIR}/libaoclsparse_portable.so"
file "${TEST_BUILD_DIR}/test_aocl_sparse"
echo ""
echo "Library:  ${INSTALL_PREFIX}/lib/libaoclsparse_portable.so"
echo "Headers:  ${INSTALL_PREFIX}/include/"
echo ""
echo "Run on target:"
echo "  LD_LIBRARY_PATH=. ./test_aocl_sparse"