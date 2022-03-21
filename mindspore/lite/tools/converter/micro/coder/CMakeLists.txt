set(3RD_DIR ${TOP_DIR}/third_party)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

if(MSLITE_ENABLE_CONVERTER AND NOT MACHINE_LINUX_ARM64)
    set(CODEGEN_PATH ${CMAKE_BINARY_DIR}/micro/coder/codegen)
else()
    set(CODEGEN_PATH ${CMAKE_BINARY_DIR}/coder/codegen)
endif()

#include 3rd
include_directories(${3RD_DIR})

#include ms
include_directories(${MICRO_DIR})
include_directories(${TOP_DIR}/)
include_directories(${TOP_DIR}/mindspore/core/)
include_directories(${NNACL_DIR}/../)
include_directories(${LITE_DIR})

#include coder
if(NOT MSVC OR NOT WIN32 OR NOT APPLE)
    include(${TOP_DIR}/cmake/external_libs/cmsis.cmake)
    include(${MICRO_DIR}/cmake/package_wrapper.cmake)
    add_subdirectory(wrapper)
endif()

if(MSLITE_ENABLE_CONVERTER)
    include(${MICRO_DIR}/cmake/file_list.cmake)
    set_property(SOURCE ${FILE_SET} PROPERTY COMPILE_OPTIONS -Wno-error=stringop-overflow=)
    add_library(coder_mid OBJECT ${FILE_SET} utils/common.h)
    add_dependencies(coder_mid fbs_src fbs_inner_src)
    target_link_libraries(coder_mid PRIVATE ${SECUREC_LIBRARY} wrapper)
endif()
