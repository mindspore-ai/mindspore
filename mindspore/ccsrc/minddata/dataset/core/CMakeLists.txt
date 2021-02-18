file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
set(DATASET_CORE_SRC_FILES
        client.cc
        config_manager.cc
        cv_tensor.cc
        data_type.cc
        device_resource.cc
        device_tensor.cc
        de_tensor.cc
        global_context.cc
        tensor.cc
        tensor_helpers.cc
        tensor_row.cc
        tensor_shape.cc
        )

if(ENABLE_ACL)
  set(DATASET_CORE_SRC_FILES
          ${DATASET_CORE_SRC_FILES}
          ascend_resource.cc)
endif()

ms_protobuf_generate(EXAMPLE_SRCS EXAMPLE_HDRS example.proto)
ms_protobuf_generate(FEATURE_SRCS FEATURE_HDRS feature.proto)
add_library(core OBJECT ${DATASET_CORE_SRC_FILES} ${EXAMPLE_SRCS} ${FEATURE_SRCS})
add_dependencies(core mindspore::protobuf)

if(ENABLE_PYTHON)
  target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})
endif()
