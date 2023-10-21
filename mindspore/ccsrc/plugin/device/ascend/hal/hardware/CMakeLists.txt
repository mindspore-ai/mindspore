file(GLOB_RECURSE HARDWARE_ASCEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

list(REMOVE_ITEM HARDWARE_ASCEND_SRC_LIST "ascend_collective_comm_lib.cc" "ascend_communication_group.cc")
set(ASCEND_COLLECTIVE_SRCS "ascend_collective_comm_lib.cc"
        "ascend_communication_group.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
set_property(SOURCE ${ASCEND_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(d_collective STATIC ${ASCEND_COLLECTIVE_SRCS})
target_link_libraries(d_collective PRIVATE ${HCCL})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

file(GLOB_RECURSE MS_HARDWARE_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend_deprecated_interface.cc"
        "ge_device_context.cc"
        "ge_device_res_manager.cc"
        "ge_graph_executor.cc"
        "ge_kernel_executor.cc"
        "ge_utils.cc"
        "ge_graph_optimization.cc"
        )

set_property(SOURCE ${HARDWARE_ASCEND_SRC_LIST} ${MS_HARDWARE_910B} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
if(ENABLE_D)
    add_library(_mindspore_plugin_device_ascend_hal_hardware_obj OBJECT ${HARDWARE_ASCEND_SRC_LIST})
    add_library(_mindspore_plugin_device_ascend_hal_hardware_910b_obj OBJECT ${MS_HARDWARE_910B})
endif()