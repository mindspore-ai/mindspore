file(GLOB_RECURSE HARDWARE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "device_context_manager.cc" "collective/*.cc")

if(ENABLE_D)
    file(GLOB_RECURSE HARDWARE_D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ascend/*.cc")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE HARDWARE_GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/*.cc")
    list(REMOVE_ITEM HARDWARE_GPU_SRC_LIST "gpu/nvidia_collective_comm_lib.cc" "gpu/nvidia_communication_group.cc")
    if(ENABLE_MPI)
        set(NVIDIA_COLLECTIVE_SRCS "gpu/nvidia_collective_comm_lib.cc"
                                "gpu/nvidia_communication_group.cc"
                                "collective/collective_communication_lib.cc"
                                "collective/communication_group.cc")
        set_property(SOURCE ${NVIDIA_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
                        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(nvidia_collective SHARED ${NVIDIA_COLLECTIVE_SRCS})
        target_link_libraries(nvidia_collective PRIVATE mindspore::nccl)
    endif()
endif()

if(ENABLE_CPU)
    file(GLOB_RECURSE HARDWARE_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "cpu/*.cc")
    list(REMOVE_ITEM HARDWARE_CPU_SRC_LIST "cpu/mpi_collective_comm_lib.cc" "cpu/mpi_communication_group.cc")
    if(WIN32)
        list(REMOVE_ITEM HARDWARE_CPU_SRC_LIST "cpu/ms_collective_comm_lib.cc")
    endif()
    if(ENABLE_MPI)
        set(MPI_COLLECTIVE_SRCS "cpu/mpi_collective_comm_lib.cc"
                                "cpu/mpi_communication_group.cc"
                                "collective/collective_communication_lib.cc"
                                "collective/communication_group.cc")
        set_property(SOURCE ${MPI_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
                        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(mpi_collective SHARED ${MPI_COLLECTIVE_SRCS})
        target_link_libraries(mpi_collective PRIVATE mindspore::ompi)
    endif()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()


set_property(SOURCE ${HARDWARE_SRC_LIST} ${HARDWARE_D_SRC_LIST} ${HARDWARE_GPU_SRC_LIST} ${HARDWARE_CPU_SRC_LIST}
                    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_hardware_obj OBJECT ${HARDWARE_SRC_LIST} ${HARDWARE_D_SRC_LIST}
            ${HARDWARE_GPU_SRC_LIST} ${HARDWARE_CPU_SRC_LIST})
