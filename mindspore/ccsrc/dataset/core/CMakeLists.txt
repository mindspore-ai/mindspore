add_library(core OBJECT
  client.cc
  config_manager.cc
  cv_tensor.cc
  data_type.cc
  global_context.cc
  tensor.cc
  tensor_shape.cc
  )
target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})
