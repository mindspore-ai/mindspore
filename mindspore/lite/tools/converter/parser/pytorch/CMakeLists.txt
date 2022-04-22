include(${TOP_DIR}/mindspore/lite/cmake/merge.cmake)
merge_parser(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tools/converter/parser/pytorch/pytorch_op_parser.cc)
file(GLOB_RECURSE PYTORCH_SRC_LIST ${CMAKE_BINARY_DIR}/tools/converter/parser/pytorch/pytorch_op_parser.cc)
set_property(SOURCE ${PYTORCH_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(pytorch_parser_mid OBJECT
        ${PYTORCH_SRC_LIST}
        )

add_compile_definitions(C10_USE_GLOG)
if(NOT EXISTS ${LIB_TORCH_PATH})
    message(FATAL_ERROR "Path of libtorch is invalid.")
endif()
find_package(Torch REQUIRED PATHS ${LIB_TORCH_PATH})
if(TORCH_FOUND)
    target_link_libraries(pytorch_parser_mid PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(pytorch_parser_mid PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Torch is not found")
endif()

add_dependencies(pytorch_parser_mid fbs_src)
add_dependencies(pytorch_parser_mid fbs_inner_src)
