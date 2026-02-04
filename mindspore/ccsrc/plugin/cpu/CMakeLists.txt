if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwins")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

if(ENABLE_CPU)
    file(GLOB_RECURSE _CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM _CPU_SRC_LIST
        "res_manager/collective/mpi_collective_comm_lib.cc"
        "res_manager/collective/mpi_communication_group.cc")
    # Windows can not register MakeCPUDeviceAddress automatically.
    if(WIN32)
        list(REMOVE_ITEM _CPU_SRC_LIST "cpu_device_address_maker.cc")
    endif()
    if(WIN32 OR APPLE)
        list(REMOVE_ITEM _CPU_SRC_LIST
            "res_manager/collective/ms_collective_comm_lib.cc"
            "res_manager/collective/allreduce_impl.cc"
            "res_manager/collective/ms_collective_topo.cc"
            "res_manager/collective/ms_collective_node.cc")
    endif()

    if(ENABLE_MPI)
        set(_CPU_MPI_COLLECTIVE_SRC_LIST
            "res_manager/collective/mpi_collective_comm_lib.cc"
            "res_manager/collective/mpi_communication_group.cc"
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
        set_property(SOURCE ${_CPU_MPI_COLLECTIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(mpi_collective SHARED ${_CPU_MPI_COLLECTIVE_SRC_LIST})
        target_link_libraries(mpi_collective PRIVATE mindspore::ompi)
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
    endif()

    set_property(SOURCE ${_CPU_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(mindspore_cpu SHARED ${_CPU_SRC_LIST})
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_cpu PRIVATE CPU_DLL)
    endif()
    target_link_libraries(mindspore_cpu PRIVATE mindspore_core mindspore_ops mindspore_common mindspore_ops_cpu
        mindspore_backend_common mindspore_hardware_abstract mindspore_profiler mindspore_memory_pool proto_input
        mindspore_runtime_utils mindspore::dnnl mindspore::pybind11_module
        mindspore::securec mindspore_ops_host)
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set_target_properties(mindspore_cpu PROPERTIES INSTALL_RPATH @loader_path:@loader_path/plugin)
    else()
        set_target_properties(mindspore_cpu PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/plugin)
    endif()
endif()
