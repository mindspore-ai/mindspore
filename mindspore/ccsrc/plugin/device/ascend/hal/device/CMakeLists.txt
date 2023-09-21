if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(DEFINED ENV{ENABLE_ASYNC_LINK} AND
        ("$ENV{ENABLE_ASYNC_LINK}" STREQUAL "ON" OR "$ENV{ENABLE_ASYNC_LINK}" STREQUAL "on"))
    add_compile_definitions(ENABLE_ASYNC_LINK)
endif()

file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
list(REMOVE_ITEM DEVICE_SRC_LIST "distribute/mpi_collective_group.cc"
        "distribute/collective_group_wrapper.cc" "distribute/mpi_pycc.cc")

if(ENABLE_MPI AND ENABLE_D)
    set_property(SOURCE "distribute/mpi_pycc.cc"
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    pybind11_add_module(_ascend_mpi "distribute/mpi_pycc.cc")
    target_link_libraries(_ascend_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
endif()

list(REMOVE_ITEM DEVICE_SRC_LIST "profiling/profiling_callback_register.cc")

if(ENABLE_SECURITY)
    list(REMOVE_ITEM DEVICE_SRC_LIST "profiling/profiling_callback_register.cc")
    list(REMOVE_ITEM DEVICE_SRC_LIST "profiling/profiling_manager.cc")
    list(REMOVE_ITEM DEVICE_SRC_LIST "profiling/profiling_utils.cc")
endif()

set_property(SOURCE ${DEVICE_SRC_LIST}}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

set(MS_DEVICE_910B ${DEVICE_SRC_LIST})
#==========remove unused file in 910b=============
file(GLOB_RECURSE UNUSED_SRC_IN_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./ge_runtime/*.cc"
        "./tasksink/*.cc"
        "./kernel_build_ascend.cc"
        "./kernel_select_ascend.cc"
        "./kernel_select_graph_kernel.cc"
        "./ascend_label_assign.cc"
        "./ascend_launch_mul.cc"
        "./ascend_launch_transdata.cc"
        "./ascend_launch_kernel.cc"
        "./ascend_runtime_core.cc"
        "./lic_manager.cc"
        )
list(REMOVE_ITEM MS_DEVICE_910B ${UNUSED_SRC_IN_910B})
set_property(SOURCE ${MS_DEVICE_910B} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

if(ENABLE_D)
    add_library(_mindspore_plugin_device_ascend_hal_device_obj OBJECT ${DEVICE_SRC_LIST})
    add_library(_mindspore_plugin_device_ascend_hal_device_910b_obj OBJECT ${MS_DEVICE_910B})

    file(GLOB_RECURSE GE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ge_runtime/*.cc")
    set_property(SOURCE ${GE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_GE)
    target_include_directories(_mindspore_plugin_device_ascend_hal_device_obj PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
    add_dependencies(_mindspore_plugin_device_ascend_hal_device_obj graph)
    target_include_directories(_mindspore_plugin_device_ascend_hal_device_910b_obj PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
    add_dependencies(_mindspore_plugin_device_ascend_hal_device_910b_obj graph)
    if(ENABLE_MPI)
        find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
        file(GLOB_RECURSE ASCEND_COLLECTIVE_LIST "distribute/mpi_collective_group.cc"
                "distribute/collective_group_wrapper.cc")
        set_property(SOURCE ${ASCEND_COLLECTIVE_LIST}
                PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(ascend_collective SHARED ${ASCEND_COLLECTIVE_LIST})
        target_link_libraries(ascend_collective PRIVATE ${HCCL} mindspore::ompi)
        target_link_libraries(_ascend_mpi PRIVATE ascend_collective)
    endif()
endif()
