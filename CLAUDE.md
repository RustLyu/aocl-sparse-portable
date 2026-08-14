# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AOCL-Sparse Portable — a stripped-down C++14 port of AMD's AOCL-Sparse library, cross-compiled for **ARMv7-A (32-bit, hard-float)** using GCC 6.2.1 (Linaro). The original codebase was C++17 + x86 SIMD; this port downgrades to C++14 and replaces all SIMD paths with ARM NEON (single-precision) + VFPv3 scalar fallback (double-precision).

**Key limitation:** This is a **portable/reference build** — all x86 SIMD kernels (AVX2/AVX512/KT) have been removed. The kernel dispatcher always routes to scalar reference implementations. Only CSR, CSC, COO, and TCSR matrix formats are supported.

## Build Commands

### One-shot build (recommended)
```bash
./build.sh
```
This cross-compiles the shared library, the `test_aocl_sparse` test executable, and installs everything into `$HOME/opt/arm-v7-a/aocl-sparse-portable/`.

### Manual CMake configure + build
```bash
TOOLCHAIN="$HOME/tools/petaLinux/tools/linux-i386/gcc-arm-linux-gnueabi"

cmake -S . -B build_arm_gcc \
    -DCMAKE_TOOLCHAIN_FILE=toolchain-arm-gnueabihf.cmake \
    -DCMAKE_CXX_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=hard" \
    -DOpenMP_CXX_FLAGS="-fopenmp" \
    -DOpenMP_CXX_LIB_NAMES="gomp" \
    -DOpenMP_gomp_LIBRARY="${TOOLCHAIN}/arm-linux-gnueabihf/lib/libgomp.so" \
    -DENABLE_ARM_SIMD=OFF \
    -DSUPPORT_OMP=ON

make -C build_arm_gcc -j$(nproc)
```

### Build test only (requires library already installed)
```bash
cmake -S test_spmm -B test_spmm/build \
    -DCMAKE_TOOLCHAIN_FILE=toolchain-arm-gnueabihf.cmake \
    -DCMAKE_C_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=hard"
make -C test_spmm/build -j$(nproc)
```

## Architecture

### Source layout
```
include/           # Public C API headers (installed)
  aoclsparse.h           # Umbrella header
  aoclsparse_auxiliary.h # Matrix create/destroy/export, descriptor mgmt
  aoclsparse_functions.h # Level-2 (SpMV, SpSV), Level-3 (SpMM, Sp2M, SpMMD), solvers
  aoclsparse_types.h     # All enums, typedefs, aoclsparse_int, complex types

src/
  include/         # Private C++ headers (not installed)
    aoclsparse_neon.hpp          # NEON SIMD wrappers for float + VFPv3 fallback for double
    aoclsparse_context.hpp       # Singleton context (ISA hints, thread count, env vars)
    aoclsparse_utils.hpp         # Numeric helpers (zero, one, NaN, infinity, tolerance)
    aoclsparse_mat_structures.hpp # Internal matrix classes (csr, coo, tcsr, base_mtx)
    aoclsparse_mtx_dispatcher.hpp # Kernel dispatch tables (stubbed out in portable build)
    aoclsparse_descr.h           # Matrix descriptor struct (_aoclsparse_mat_descr)
    aoclsparse_error_check.hpp   # Input validation helpers
  auxiliary/
    aoclsparse_auxiliary.{hpp,cpp}  # Matrix create/destroy C wrappers, ISA hints, debug
  analysis/
    aoclsparse_csr_util.{hpp,cpp}   # CSR validation/sorting/conversion utilities
  conversion/
    aoclsparse_convert.hpp          # Format conversion templates (COO→CSR, etc.)
  level2/
    aoclsparse_mv.cpp               # SpMV entry point (thin wrapper)
    aoclsparse_csrmv.hpp            # SpMV scalar reference kernels + NEON fast-paths
  level3/
    aoclsparse_csrmm.cpp            # SpMM entry point (thin wrapper)
    aoclsparse_csrmm.hpp            # SpMM scalar reference kernels + NEON fast-paths
    aoclsparse_csr2m.cpp            # Sparse×sparse multiplication
    aoclsparse_csradd.{hpp,cpp}     # Sparse matrix addition
    aoclsparse_sp2m.cpp             # Sparse×dense (row-major)
    aoclsparse_sp2md.{hpp,cpp}      # Sparse×sparse→dense
```

### Pattern: template kernels + C API wrappers

Every public function follows this pattern:
1. A C++ template `aoclsparse_xxx_t<T>()` in a `.hpp` file (the real implementation).
2. A `extern "C"` thin wrapper in the corresponding `.cpp` that instantiates the template for all 4 types (`float`, `double`, `aoclsparse_float_complex`, `aoclsparse_double_complex`).

Example for `aoclsparse_create_scsr`:
- Template: `aoclsparse_create_csr_t<T>()` in `src/auxiliary/aoclsparse_auxiliary.cpp`
- C wrappers: `aoclsparse_create_scsr()`, `aoclsparse_create_dcsr()`, etc. call `aoclsparse_create_csr_t<T>()`.

### Matrix type layout

The `aoclsparse_matrix` is an opaque handle (`_aoclsparse_matrix*`). Internally it holds a `std::vector<base_mtx*>` named `mats`, where the first element is the primary representation. CSC matrices are stored internally as CSR with swapped dimensions and `doid::gt` (general-transpose). This is important: `A->mats[0]->mat_type` is always `aoclsparse_csr_mat` for CSR/CSC.

## NEON SIMD: float vs double handling

Target toolchain (GCC 6.2.1, ARMv7-A) supports `float32x4_t` but NOT `float64x2_t` (no `__ARM_FEATURE_FP64_VECTOR_ARITHMETIC`).

- **Float NEON:** Full NEON SIMD via `aoclsparse_neon::` functions — `loadu`, `storeu`, `setzero`, `set1`, `mul`, `fmadd`, `hsum`, `gather`, `prefetch_gather`.
- **Double NEON:** Scalar VFPv3 fallback via `_f64`-suffixed functions — `setzero_f64()`, `set1_f64()`, `mul_f64()`, `fmadd_f64()`, `hsum_f64()`, `gather_f64()`, `prefetch_gather_f64()`. Only `loadu`/`storeu` are overloaded (same name, different parameter types).
- A custom `float64x2_t` struct (`double val[2]`) is defined globally in `aoclsparse_neon.hpp` when `__ARM_FEATURE_FP64_VECTOR_ARITHMETIC` is absent.

## C++17 → C++14 downgrade rules

The codebase was downgraded from C++17 to C++14 for GCC 6.2.1 compatibility. When modifying or adding code, follow these rules:

| C++17 feature | C++14 replacement |
|---|---|
| `if constexpr` | Regular `if` |
| `std::is_same_v<T, U>` | `std::is_same<T, U>::value` |
| Fold expressions (`... && ...`) | Recursive variadic templates |
| `inline constexpr` variables | `constexpr` (no `inline`) |

**Critical consequence of `if constexpr` → `if`:** NEON blocks inside `if(std::is_same<T, float>::value)` compile for ALL types (including `complex<float>`/`complex<double>`). Pointer arguments must use `reinterpret_cast<const float*>(ptr)` or `reinterpret_cast<const double*>(ptr)` so the complex instantiations don't fail type-checking. See `aoclsparse_csrmv.hpp` and `aoclsparse_csrmm.hpp` for the pattern.

## Missing pieces added in this port

The portable build was missing implementations that existed in the full x86 build. These were added:

1. **`aoclsparse_create_scsr/dcsr/ccsr/zcsr`** — CSR matrix creation C wrappers + `aoclsparse_create_csr_t<T>()` template (in `src/auxiliary/aoclsparse_auxiliary.cpp`).
2. **`aoclsparse::context::get_context()`** — Singleton context (in `src/auxiliary/aoclsparse_auxiliary.cpp`).
3. **`tl_isa_hint`** — Thread-local ISA hint definition (in `src/auxiliary/aoclsparse_auxiliary.cpp`).

## Install & downstream usage

After running `./build.sh`, the library is installed at:
```
$HOME/opt/arm-v7-a/aocl-sparse-portable/
├── include/            # Public headers
├── lib/
│   ├── libaoclsparse_portable.so
│   └── cmake/aoclsparse_portable/
│       └── aoclsparse_portable-config.cmake
```

Downstream projects use `find_package`:
```cmake
find_package(aoclsparse_portable REQUIRED
    PATHS /home/lsk/opt/arm-v7-a/aoclsparse-portable
    NO_DEFAULT_PATH
)
target_link_libraries(myapp aoclsparse_portable::aoclsparse_portable)
```

The imported target automatically adds `-fopenmp` and links `gomp`.
