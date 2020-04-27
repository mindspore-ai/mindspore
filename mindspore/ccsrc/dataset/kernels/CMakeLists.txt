add_subdirectory(image)
add_subdirectory(data)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(kernels OBJECT
    py_func_op.cc
    tensor_op.cc)
target_include_directories(kernels PRIVATE ${pybind11_INCLUDE_DIRS})
