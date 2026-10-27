set(ASCEND_COLLECTIVE_SRCS "ascend_collective_comm_lib.cc"
        "ascend_communication_group.cc"
        "utils.cc"
        "dummy_ascend_collective_comm_lib.cc"
        "ascend_comm_manager.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/dummy_collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
set_property(SOURCE ${ASCEND_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(d_collective SHARED ${ASCEND_COLLECTIVE_SRCS})
find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH})
target_link_libraries(d_collective PRIVATE ${HCCL} mindspore_cluster)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)
    link_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/lib)
    set(LOWLATENCY_COLLECTIVE_SRCS "lowlatency_collective_comm_lib.cc"
            "lowlatency_communication_group.cc"
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
    set_property(SOURCE ${LOWLATENCY_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(lowlatency_collective SHARED ${LOWLATENCY_COLLECTIVE_SRCS})
    target_link_libraries(lowlatency_collective PRIVATE lcal_ms mindspore_core)
    set_target_properties(lowlatency_collective PROPERTIES INSTALL_RPATH
                    ${ORIGIN_PATH}/ms_kernels_internal/asdops/lib)
    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
endif()

set(DVM_COLLECTIVE_SRCS "dvm_collective_comm_lib.cc"
        "dvm_communication_group.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
set_property(SOURCE ${DVM_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(dvm_collective STATIC ${DVM_COLLECTIVE_SRCS})
target_link_libraries(dvm_collective PRIVATE ${DVM_LIB})
add_dependencies(dvm_collective dvm)

set(MULTI_ASCEND_COLLECTIVE_SRCS "multi_ascend_collective_comm_lib.cc"
        "multi_ascend_communication_group.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_comm_lib_loader.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
set_property(SOURCE ${MULTI_ASCEND_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(multi_ascend_collective STATIC ${MULTI_ASCEND_COLLECTIVE_SRCS})
target_link_libraries(multi_ascend_collective PRIVATE d_collective PRIVATE dvm_collective)
