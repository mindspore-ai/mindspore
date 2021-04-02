if(NOT PLATFORM_ARM64 AND NOT PLATFORM_ARM32)
    add_definitions(-DUSE_GLOG)
endif()

set(3RD_DIR ${TOP_DIR}/third_party)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(MICRO_DIR ${LITE_DIR}/micro)

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
include_directories(${TOP_DIR}/mindspore/core/)
include_directories(${LITE_DIR})
include_directories(${MICRO_DIR})
#include coder
include(${TOP_DIR}/cmake/external_libs/cmsis.cmake)
include(${MICRO_DIR}/cmake/file_list.cmake)
include(${MICRO_DIR}/cmake/package_wrapper.cmake)
add_subdirectory(wrapper)

if(ENABLE_CONVERTER)
    add_executable(codegen main.cc ${FILE_SET})
    add_dependencies(codegen fbs_src)
    add_dependencies(codegen fbs_inner_src)
    target_link_libraries(codegen PRIVATE ${SECUREC_LIBRARY} mindspore::glog)
    if(NOT WIN32 AND "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        add_custom_command(TARGET codegen POST_BUILD COMMAND strip ${CODEGEN_PATH})
    endif()
endif()
