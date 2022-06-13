if(ENABLE_CPU)
    file(GLOB_RECURSE HARDWARE_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM HARDWARE_CPU_SRC_LIST "mpi_collective_comm_lib.cc" "mpi_communication_group.cc")

    if(WIN32 OR APPLE)
        list(REMOVE_ITEM HARDWARE_CPU_SRC_LIST "ms_collective_comm_lib.cc" "allreduce_impl.cc"
          "ms_collective_ops_impl.cc")
        list(REMOVE_ITEM HARDWARE_CPU_SRC_LIST "ms_collective_topo.cc" "ms_collective_node.cc")
    endif()
    if(ENABLE_MPI)
        set(MPI_COLLECTIVE_SRCS "mpi_collective_comm_lib.cc"
                                "mpi_communication_group.cc"
                                ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
                                ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
    set_property(SOURCE ${MPI_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(mpi_collective SHARED ${MPI_COLLECTIVE_SRCS})
    target_link_libraries(mpi_collective PRIVATE mindspore::ompi)
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
    endif()

    set_property(SOURCE ${HARDWARE_CPU_SRC_LIST} PROPERTY
            COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_device_cpu_hal_hardware_obj OBJECT ${HARDWARE_CPU_SRC_LIST})
endif()
