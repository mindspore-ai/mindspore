set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections ")
set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(3RD_DIR ${TOP_DIR}/third_party)
set(NNACL_DIR ${LITE_DIR}/nnacl)

#include 3rd
include_directories(${3RD_DIR})
include_directories(${3RD_DIR}/flatbuffers/include)
#include ms
include_directories(${TOP_DIR}/)
include_directories(${LITE_DIR})
include_directories(${TOP_DIR}/mindspore/core/)

#include coder
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include(${MICRO_DIR}/cmake/file_list.cmake)
add_executable(codegen main.cc ${FILE_SET})
add_dependencies(codegen fbs_src)
add_dependencies(codegen fbs_inner_src)
target_link_libraries(codegen ${SECUREC_LIBRARY})
