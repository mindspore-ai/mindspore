set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(DEFINED ENV{HISI_TOOLCHAIN_PATH})
    set(TOOLCHAIN_PATH $ENV{HISI_TOOLCHAIN_PATH}/hisi-linux/x86_arm)
else()
    set(TOOLCHAIN_PATH "/opt/hisi-linux/x86-arm")
endif()
set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/arm-himix200-linux/bin/arm-himix200-linux-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/arm-himix200-linux/bin/arm-himix200-linux-g++)

find_path(GCC_PATH gcc)
find_path(GXX_PATH g++)
if(NOT ${GCC_PATH} STREQUAL "GCC_PATH-NOTFOUND" AND NOT ${GXX_PATH} STREQUAL "GXX_PATH-NOTFOUND")
    set(FLATC_GCC_COMPILER ${GCC_PATH}/gcc)
    set(FLATC_GXX_COMPILER ${GXX_PATH}/g++)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CROSS_COMPILATION_ARM himix200)
set(CROSS_COMPILATION_ARCHITECTURE armv7-a)

set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4  ${CMAKE_CXX_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

set(HIMIX_STRIP ${TOOLCHAIN_PATH}/arm-himix200-linux/bin/arm-himix200-linux-strip)
