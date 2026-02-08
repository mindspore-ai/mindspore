if(ENABLE_D OR ENABLE_ACL)
  file(GLOB _ASCEND_RES_MANAGER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  set_property(SOURCE ${_ASCEND_RES_MANAGER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
  if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    include_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    link_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/shmem/shmem/lib)
    link_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/shmem/memfabric_hybrid/lib)
    add_compile_definitions(ENABLE_SYMMETRIC_MEMORY_KERNELS)
  endif()
  add_library(_mindspore_ascend_res_manager_obj OBJECT ${_ASCEND_RES_MANAGER_SRC_LIST})

  add_subdirectory(op_adapter)
  add_subdirectory(symbol_interface)
  add_subdirectory(device_context_conf)
  add_subdirectory(stream_manager)
  add_subdirectory(hal_manager)
  add_subdirectory(mbuf_manager)
  add_subdirectory(error_manager)
  add_subdirectory(event)
  add_subdirectory(mem_manager)
  add_subdirectory(hccl_adapter)
  add_subdirectory(capture_graph)
  add_subdirectory(data_queue)
  add_subdirectory(allocator)
  add_subdirectory(hdk_uvm)
  add_subdirectory(symmetric_memory)

  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_op_adapter_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_symbol_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_context_conf_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_stream_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_hal_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_event_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_capture_graph_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_mem_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_hccl_adapter_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_mbuf_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_error_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_res_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_data_queue_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_allocator_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_hdk_uvm_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_symmetric_memory_manager_obj>)

  set(MS_SYMMETRIC_MEMORY_LIB_PATH
    ${ORIGIN_PATH}/ms_kernels_symmetric_memory/shmem/shmem/lib)
  set(MS_SYMMETRIC_MEMORY_LIB_PATH
    ${ORIGIN_PATH}/ms_kernels_symmetric_memory/shmem/memfabric_hybrid/lib:${MS_SYMMETRIC_MEMORY_LIB_PATH})

  add_library(mindspore_ascend_res_manager SHARED ${ASCEND_RES_MANAGER_OBJECTS_SRC})
  target_link_libraries(mindspore_ascend_res_manager PRIVATE mindspore_core mindspore_common pybind11::module
    proto_input mindspore_profiler mindspore_hardware_abstract mindspore_runtime_utils)

  if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    target_link_libraries(mindspore_ascend_res_manager PRIVATE shmem)
  endif()

  set_target_properties(mindspore_ascend_res_manager PROPERTIES
    INSTALL_RPATH $ORIGIN:$ORIGIN/../..:${MS_SYMMETRIC_MEMORY_LIB_PATH})
endif()

