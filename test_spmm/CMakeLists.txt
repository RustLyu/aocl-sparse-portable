cmake_minimum_required(VERSION 3.13)
project(test_spmm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ----- Option A: find_package (requires install first) ----------
find_package(aoclsparse_portable REQUIRED
    PATHS /home/lsk/opt/arm-v7-a/aocl-sparse-portable
    NO_DEFAULT_PATH
)

# ----- Option B: direct paths (no install needed) ---------------
# set(AOCLSPARSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
# include_directories(${AOCLSPARSE_DIR}/include
#                     ${AOCLSPARSE_DIR}/build_arm_gcc/include)
# add_executable(test_aocl_sparse test_spmm.c)
# target_link_libraries(test_aocl_sparse ${AOCLSPARSE_DIR}/build_arm_gcc/libaoclsparse_portable.so)

# ----- Build ----------------------------------------------------
add_executable(test_aocl_sparse test_spmm.c)
target_link_libraries(test_aocl_sparse aoclsparse_portable::aoclsparse_portable)
