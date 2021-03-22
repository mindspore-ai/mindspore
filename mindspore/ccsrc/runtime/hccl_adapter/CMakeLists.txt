file(GLOB_RECURSE HCCL_ADAPTER_SRC_LIST ./*.cc)
set_property(SOURCE ${HCCL_ADAPTER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_HCCL_ADPT)
if(ENABLE_D)
    add_library(_mindspore_runtime_hccl_adapter_obj OBJECT ${HCCL_ADAPTER_SRC_LIST})
    target_include_directories(_mindspore_runtime_hccl_adapter_obj PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
    add_dependencies(_mindspore_runtime_hccl_adapter_obj graph)
endif()