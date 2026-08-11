cmake_minimum_required(VERSION 3.13)
project(test_spmm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../build_arm_gcc/include
)

add_executable(test_spmm test_spmm.c)
target_link_libraries(test_spmm ${CMAKE_CURRENT_SOURCE_DIR}/../build_arm_gcc/libaoclsparse_portable.so)
