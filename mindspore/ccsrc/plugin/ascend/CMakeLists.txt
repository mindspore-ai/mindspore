include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)
endif()

if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    add_compile_definitions(ENABLE_SYMMETRIC_MEMORY_KERNELS)
    include_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    link_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/shmem/shmem/lib)
    link_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/shmem/memfabric_hybrid/lib)
endif()

########### mindspore_ascend.so #####
file(GLOB_RECURSE ASCEND_910B_OBJ_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_device_context.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_device_address_maker.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/graph_optimizer/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/kernel_executor/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/profiler/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/stress_detect/*.cc")

set_property(SOURCE ${ASCEND_910B_OBJ_SRC}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(mindspore_ascend SHARED ${ASCEND_910B_OBJ_SRC})
set_target_properties(mindspore_ascend PROPERTIES VERSION 2)

set(MS_KERNELS_INTERNAL_LIB_PATH
    ${ORIGIN_PATH}/ascend/ms_kernels_internal/internal_kernel/lib)

set(MS_SM_LIB
    ${ORIGIN_PATH}/ms_kernels_symmetric_memory/shmem/shmem/lib:)
set(MS_SM_LIB
    ${ORIGIN_PATH}/ms_kernels_symmetric_memory/shmem/memfabric_hybrid/lib:${MS_SM_LIB})

target_link_libraries(mindspore_ascend PRIVATE mindspore_backend_common mindspore_core mindspore_ops mindspore_common
                      mindspore_ops_ascend mindspore_profiler mindspore_runtime_pipeline
                      mindspore_hardware_abstract mindspore_runtime_utils mindspore_cpu mindspore_ops_host)
target_link_libraries(mindspore_ascend PRIVATE mindspore_ascend_res_manager)
target_link_libraries(mindspore_ascend PRIVATE -Wl,--no-as-needed mindspore_pyboost -Wl,--as-needed)
target_link_libraries(mindspore_ascend PRIVATE proto_input mindspore::protobuf)
target_link_libraries(mindspore_ascend PRIVATE mindspore::securec d_collective)
set_target_properties(mindspore_ascend PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/ascend:${ORIGIN_PATH}/../:${MS_KERNELS_INTERNAL_LIB_PATH}:${MS_SM_LIB})

include(${CMAKE_SOURCE_DIR}/cmake/dependency_metadef.cmake)

MESSAGE("USE DAV LIB PATH: ${ASCEND_PATH}")
find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH})
find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH})
find_library(ACL ascendcl ${ASCEND_CANN_RUNTIME_PATH})
find_library(PROFILING msprofiler ${ASCEND_CANN_RUNTIME_PATH})

target_link_libraries(mindspore_ascend PRIVATE -Wl,--no-as-needed ${GE_RUNNER} ${GRAPH} ${ACL} ${PROFILING})
