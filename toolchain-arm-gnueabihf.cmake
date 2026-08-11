# ARM Linux gnueabihf cross-compilation toolchain
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7-a)

set(TOOLCHAIN_PREFIX "$ENV{HOME}/tools/petaLinux/tools/linux-i386/gcc-arm-linux-gnueabi")

set(CMAKE_C_COMPILER    "${TOOLCHAIN_PREFIX}/bin/arm-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER  "${TOOLCHAIN_PREFIX}/bin/arm-linux-gnueabihf-g++")
set(CMAKE_AR             "${TOOLCHAIN_PREFIX}/bin/arm-linux-gnueabihf-ar" CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB         "${TOOLCHAIN_PREFIX}/bin/arm-linux-gnueabihf-ranlib" CACHE FILEPATH "Ranlib")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)