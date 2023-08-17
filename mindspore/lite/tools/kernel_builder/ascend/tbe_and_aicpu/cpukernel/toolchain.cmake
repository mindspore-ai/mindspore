#this is required, only this variable is set, CMAKE recognize this is cross compilation
#in addition, CMAKE_CROSSCOMPILING is set true when CMAKE_SYSTEM_NAME is set
set(CMAKE_SYSTEM_NAME Linux)

#Change the path to the absolute path of the cross compilation tool after the toolkit package is decompressed
# set cross compile toolchain dir
set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
#Specify cross compiler
if(LMIX)
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}-g++)
        set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}-gcc)
else()
    if(NOT "x${TOOLCHAIN_DIR}" STREQUAL "x")
        if(NOT IS_DIRECTORY ${TOOLCHAIN_DIR})
            message(FATAL_ERROR "specify cross compile toolchain directory(${TOOLCHAIN_DIR}) is not exist")
        endif()
    endif()
    message(STATUS "TOOLCHAIN_DIR=${TOOLCHAIN_DIR}")
    if(MINRC)
        set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-g++)
    else()
        set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
    endif()
endif()
#For FIND_PROGRAM(), there are three values, NEVER, ONLY, BOTH
#the first means not to search under your CMAKE_FIND_ROOT_PATH
#the second means to search only under this path
#the third means to find this path first, then Find the global path.
#For this variable, it is usually a program that calls the host, so it is generally set to NEVER
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

#The following three options indicate that only libraries and header files are found in the cross environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
