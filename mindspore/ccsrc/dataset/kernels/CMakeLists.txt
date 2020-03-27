add_subdirectory(image)
add_subdirectory(data)
add_library(kernels OBJECT
    py_func_op.cc
    tensor_op.cc)
target_include_directories(kernels PRIVATE ${pybind11_INCLUDE_DIRS})
