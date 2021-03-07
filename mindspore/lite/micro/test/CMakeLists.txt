add_definitions(-DUSE_GLOG)
string(REPLACE "/test" "" MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR})
string(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(3RD_DIR ${TOP_DIR}/third_party)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(BUILD_LITE "on")

include(${TOP_DIR}/cmake/external_libs/gtest.cmake)
include(${MICRO_DIR}/cmake/file_list.cmake)
include(${MICRO_DIR}/cmake/package_wrapper.cmake)

include_directories(${TOP_DIR})
include_directories(${TOP_DIR}/mindspore/core/)
include_directories(${LITE_DIR})
include_directories(${MICRO_DIR})
include_directories(${3RD_DIR})

add_executable(micro_test code_gen_test.cc ${FILE_SET})
add_dependencies(micro_test fbs_src)
add_dependencies(micro_test fbs_inner_src)
target_link_libraries(micro_test dl mindspore::gtest ${SECUREC_LIBRARY} mindspore::glog)
