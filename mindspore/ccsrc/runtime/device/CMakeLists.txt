file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc" "gsm/*.cc"
    "kernel_info.cc" "executor/dynamic_kernel.cc" "executor/executor_callback.cc" "kernel_runtime.cc"
    "memory_manager.cc" "kernel_runtime_manager.cc" "convert_tensor_utils.cc" "memory_scheduler.cc"
    "memory_offload_strategy.cc" "launch_kernel.cc" "launch_mul.cc" "tensor_array.cc"
    "ms_device_shape_transfer.cc" "context_extends.cc" "stream_synchronizer.cc" "tensors_queue.cc" "auto_mem_offload.cc"
    "common_somas_allocator.cc" "device_address_utils.cc" "loadable_device_address.cc"
)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_device_obj OBJECT ${DEVICE_SRC_LIST})
