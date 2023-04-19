set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

find_program(arm-none-eabi-gcc_EXE arm-none-eabi-gcc)
if(NOT arm-none-eabi-gcc_EXE)
    message(FATAL_ERROR "Required C COMPILER arm-none-eabi-gcc not found, "
            "please install the package and try building MindSpore again.")
else()
    message("Find C COMPILER PATH: ${arm-none-eabi-gcc_EXE}")
endif()

find_program(arm-none-eabi-g++_EXE arm-none-eabi-g++)
if(NOT arm-none-eabi-g++_EXE)
    message(FATAL_ERROR "Required CXX COMPILER arm-none-eabi-g++ not found, "
            "please install the package and try building MindSpore again.")
else()
    message("Find CXX COMPILER PATH: ${arm-none-eabi-g++_EXE}")
endif()

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)

set(CMAKE_CXX_FLAGS "-mcpu=cortex-m7 -fstack-protector-strong -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
    -specs=nosys.specs -specs=nano.specs")
set(CMAKE_C_FLAGS "-mcpu=cortex-m7 -fstack-protector-strong -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
    -specs=nosys.specs -specs=nano.specs")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CROSS_COMPILATION_ARM contex-m7)
set(CROSS_COMPILATION_ARCHITECTURE armv7-m)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

SET(CMAKE_C_COMPILER_WORKS TRUE)
SET(CMAKE_CXX_COMPILER_WORKS TRUE)
SET(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
