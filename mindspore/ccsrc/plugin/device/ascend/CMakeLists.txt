include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/lib64)
set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/lib64)
set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/latest/lib64)
set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling)
set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
set(MINDSPORE_RPATH
        ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)

########### mindspore_ascend.so #####
set(ASCEND_SUB_COMP
        hal/common
        hal/device
        hal/hardware
        hal/hccl_adapter
        hal/profiler
        kernel
        optimizer
        )

foreach(a_comp ${ASCEND_SUB_COMP})
    add_subdirectory(${a_comp})
    string(REPLACE "/" "_" sub ${a_comp})
    if(TARGET _mindspore_plugin_device_ascend_${sub}_obj)
        list(APPEND ASCEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_ascend_${sub}_obj>)
        add_dependencies(_mindspore_plugin_device_ascend_${sub}_obj proto_input)
    endif()
endforeach()

list(APPEND ASCEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_transform_graph_ir_obj>)
add_library(mindspore_ascend SHARED ${ASCEND_SUB_OBJECTS_SRC})
target_link_libraries(mindspore_ascend PRIVATE mindspore_backend mindspore_core mindspore_common)
target_link_libraries(mindspore_ascend PRIVATE proto_input mindspore::protobuf)
target_link_libraries(mindspore_ascend PRIVATE securec)
target_link_libraries(mindspore_ascend PRIVATE mindspore::dnnl mindspore::mkldnn mindspore::ssl
                      mindspore::crypto nnacl)
set_target_properties(mindspore_ascend PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(ENABLE_D)
        target_link_libraries(mindspore_ascend PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(ENABLE_D)
    find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_ascend PRIVATE ${GE_RUNNER} ${GRAPH} ${HCCL})
    target_link_libraries(mindspore_ascend PRIVATE mindspore::event mindspore::event_pthreads
            mindspore::event_openssl -Wl,--no-as-needed mindspore::event_core ps_cache)
endif()

if(MODE_ASCEND_ALL)
    MESSAGE("USE DAV LIB PATH: ${ASCEND_PATH}")
    find_library(ERROR_MANAGER error_manager ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(RUNTIME_LIB runtime ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(TSDCLIENT tsdclient HINTS ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(DATATRANSFER datatransfer HINTS ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(PROFILING msprofiler ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ACL ascendcl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ACL_TDT_CHANNEL acl_tdt_channel ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(PLATFORM platform ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(OPT_FEATURE opt_feature ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(adump_server libadump_server.a ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(OPTILING optiling ${ASCEND_CANN_OPP_PATH} ${ASCEND_TOOLKIT_OPP_PATH})
    find_library(ACL_OP_COMPILER acl_op_compiler ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

    target_link_libraries(mindspore_ascend PRIVATE ${RUNTIME_LIB} ${TSDCLIENT} ${DATATRANSFER} ${ERROR_MANAGER}
            -Wl,--no-as-needed ${OPTILING} ${PLATFORM} ${ACL} ${ACL_TDT_CHANNEL} ${OPT_FEATURE} ${PROFILING}
            ${ACL_OP_COMPILER})
    target_link_libraries(mindspore_ascend PRIVATE ${adump_server})
endif()

if(ENABLE_D)
    if(ENABLE_MPI)
        set_target_properties(_ascend_mpi PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernel/aicpu/aicpu_ops)
add_subdirectory(kernel/aicpu/aicpu_ops)
