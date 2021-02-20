set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections ")
set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(3RD_DIR ${TOP_DIR}/third_party)
if(ENABLE_CONVERTER)
    set(CODEGEN_PATH ${CMAKE_BINARY_DIR}/micro/coder/codegen)
else()
    set(CODEGEN_PATH ${CMAKE_BINARY_DIR}/coder/codegen)
endif()

#include 3rd
include_directories(${3RD_DIR})
include_directories(${3RD_DIR}/flatbuffers/include)
#include ms
include_directories(${TOP_DIR}/)
include_directories(${LITE_DIR})
include_directories(${TOP_DIR}/mindspore/core/)

#include coder
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include(${MICRO_DIR}/cmake/file_list.cmake)
include(${MICRO_DIR}/cmake/wrapper.cmake)
add_executable(codegen main.cc ${FILE_SET})
add_dependencies(codegen fbs_src)
add_dependencies(codegen fbs_inner_src)
target_link_libraries(codegen PRIVATE ${SECUREC_LIBRARY})
if(NOT WIN32)
  add_custom_command(TARGET codegen POST_BUILD COMMAND strip ${CODEGEN_PATH})
endif()
