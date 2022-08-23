if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_CPU)
    file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM DEVICE_SRC_LIST "mpi/mpi_adapter.cc" "mpi/mpi_export.cc")
endif()

if(ENABLE_MPI AND ENABLE_CPU)
    file(GLOB_RECURSE MPI_SRC_LIST "mpi/mpi_adapter.cc" "mpi/mpi_export.cc")
    set_property(SOURCE ${MPI_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(mpi_adapter SHARED ${MPI_SRC_LIST})
    target_link_libraries(mpi_adapter PRIVATE mindspore::ompi mindspore::pybind11_module -ldl ${SECUREC_LIBRARY})
endif()

set_property(SOURCE ${DEVICE_SRC_LIST}}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
if(ENABLE_CPU)
    add_library(_mindspore_plugin_device_cpu_hal_device_obj OBJECT ${DEVICE_SRC_LIST})
endif()