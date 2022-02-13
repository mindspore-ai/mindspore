if(ENABLE_GPU)
    file(GLOB_RECURSE HARDWARE_GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM HARDWARE_GPU_SRC_LIST "nvidia_collective_comm_lib.cc" "nvidia_communication_group.cc")
    if(ENABLE_MPI)
        set(NVIDIA_COLLECTIVE_SRCS "nvidia_collective_comm_lib.cc"
                                "nvidia_communication_group.cc"
                                ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
                                ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
        set_property(SOURCE ${NVIDIA_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
                SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(nvidia_collective SHARED ${NVIDIA_COLLECTIVE_SRCS})
        target_link_libraries(nvidia_collective PRIVATE mindspore::nccl)
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
    endif()

    set_property(SOURCE ${HARDWARE_GPU_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

    add_library(_mindspore_plugin_device_gpu_hal_hardware_obj OBJECT ${HARDWARE_GPU_SRC_LIST})
endif()