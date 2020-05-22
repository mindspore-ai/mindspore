ms_protobuf_generate(EXAMPLE_SRCS EXAMPLE_HDRS example.proto)
ms_protobuf_generate(FEATURE_SRCS FEATURE_HDRS feature.proto)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(core OBJECT
  ${EXAMPLE_SRCS}
  ${FEATURE_SRCS}
  client.cc
  config_manager.cc
  cv_tensor.cc
  data_type.cc
  global_context.cc
  tensor.cc
  tensor_shape.cc
  )
add_dependencies(core mindspore::protobuf)
target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})
