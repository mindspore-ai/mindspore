cmake_minimum_required(VERSION 3.14)
project(micro)

option(BUILD_TESTCASES "if build testcase" on)
option(PLATFORM_ARM32 "build operator for android arm 32" off)
option(PLATFORM_ARM64 "build operator for android arm 64" off)

string(REPLACE "/mindspore/lite/micro" "" TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_BINARY_DIR})
include(${TOP_DIR}/cmake/utils.cmake)
include(${TOP_DIR}/cmake/dependency_utils.cmake)
include(${TOP_DIR}/cmake/dependency_securec.cmake)
if(NOT PLATFORM_ARM64 AND NOT PLATFORM_ARM32)
  set(ENABLE_CONVERTER ON)
  set(BUILD_LITE ON)
  include(${TOP_DIR}/cmake/external_libs/glog.cmake)
  ### flatbuffer
  include(${TOP_DIR}/cmake/external_libs/flatbuffers.cmake)
  file(GLOB FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../schema/*.fbs)
  ms_build_flatbuffers_lite(FBS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../schema/
    fbs_src
    ${CMAKE_BINARY_DIR}/schema
    ""
  )
  ms_build_flatbuffers_lite(FBS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../schema/
    fbs_inner_src
    ${CMAKE_BINARY_DIR}/schema/inner
    "inner"
  )
else()
  set(ENABLE_CONVERTER OFF)
endif()


set(CMAKE_CXX_STANDARD 17)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    MESSAGE("******Micro Debug********")
    set(CMAKE_C_FLAGS "-Wall -Werror -ftrapv -DDebug -g -fvisibility=default ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-Wall -Werror -ftrapv -DDebug -g -fvisibility=default ${CMAKE_CXX_FLAGS}")
else()
    MESSAGE(" ******Micro Release********")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror \
    -fstack-protector-all -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror \
    -fstack-protector-all -Wno-attributes -Wno-deprecated-declarations \
    -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
endif()

if(NOT WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,-z,noexecstack -fPIC -pie ${CMAKE_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(coder)
if(BUILD_TESTCASES AND NOT PLATFORM_ARM64 AND NOT PLATFORM_ARM32)
  add_subdirectory(test)
endif()
