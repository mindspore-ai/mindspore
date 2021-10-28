set(3RD_DIR ${TOP_DIR}/third_party)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)
set(MICRO_DIR ${LITE_DIR}/micro)

if(MSLITE_ENABLE_CONVERTER)
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
include_directories(${NNACL_DIR}/../)
include_directories(${LITE_DIR})
include_directories(${MICRO_DIR})

#include coder
if(NOT MSVC)
    include(${TOP_DIR}/cmake/external_libs/cmsis.cmake)
    include(${MICRO_DIR}/cmake/package_wrapper.cmake)
    add_subdirectory(wrapper)
endif()

if(MSLITE_ENABLE_CONVERTER)
    include(${MICRO_DIR}/cmake/file_list.cmake)
    set_property(SOURCE ${FILE_SET} PROPERTY COMPILE_OPTIONS -Wno-error=stringop-overflow=)
    add_executable(codegen main.cc ${FILE_SET})
    add_dependencies(codegen fbs_src)
    add_dependencies(codegen fbs_inner_src)
    target_link_libraries(codegen PRIVATE ${SECUREC_LIBRARY} wrapper_mid nnacl_mid cpu_ops_mid)
    if(ENABLE_MODEL_OBF)
        target_link_libraries(codegen PRIVATE
                ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
    endif()
endif()
