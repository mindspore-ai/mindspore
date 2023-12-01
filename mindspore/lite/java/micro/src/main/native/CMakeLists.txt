cmake_minimum_required(VERSION 3.14)
project(micro_jni)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fPIC)

set(JAVA_HOME "$ENV{JAVA_HOME}")
include_directories(${JAVA_HOME}/include/) ## jni
include_directories(${JAVA_HOME}/include/linux) ## jni
include_directories(${MICRO_BUILD_PATH})

if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

get_filename_component(PKG_PATH ${PKG_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(HEADER_PATH ${PKG_PATH}/runtime)
set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)
set(NATIVE_DIR ${MICRO_JNI_PATH})

add_compile_definitions(NOT_USE_STL)

option(PLATFORM_ARM64 "build android arm64" OFF)
option(PLATFORM_ARM32 "build android arm32" OFF)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    add_compile_definitions(ENABLE_NEON)
    add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
    add_definitions(-mfloat-abi=softfp -mfpu=neon)
    add_compile_definitions(ENABLE_ARM82_A32)
endif()

if(NOT PLATFORM_ARM32 AND NOT PLATFORM_ARM64)
    include_directories(${PKG_PATH}/tools/codegen/third_party/include)
    include_directories(${PKG_PATH}/tools/codegen/third_party/include/CMSIS/Core/Include)
    include_directories(${PKG_PATH}/tools/codegen/third_party/include/CMSIS/DSP/Include)
    include_directories(${PKG_PATH}/tools/codegen/third_party/include/CMSIS/NN/Include)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build net library with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    message(STATUS "build net library release version")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(${OP_HEADER_PATH})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
include_directories(${MICRO_BUILD_PATH}/../../../)
include_directories(${MICRO_BUILD_PATH}/../../../../core/)

file(GLOB NATIVE_SRC ${NATIVE_DIR}/*.cpp)

add_library(libnet
        STATIC
        IMPORTED)

set_target_properties(libnet
        PROPERTIES IMPORTED_LOCATION
        ${MICRO_STATIC_LIB})

add_library(micro_jni SHARED
        ${MICRO_BUILD_PATH}/../../../src/common/log.cc
        ${NATIVE_SRC}
        )

target_link_libraries(micro_jni
        libnet
        )

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    find_library(log-lib log)
    target_link_libraries(micro_jni
            ${log-lib}
            )
endif()
