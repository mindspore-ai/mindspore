include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)

set(ASCEND_RPATH ${ASCEND_RPATH}:/usr/local/Ascend/nnae/latest/lib64)
set(ASCEND_RPATH ${ASCEND_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/lib64)
set(ASCEND_RPATH ${ASCEND_RPATH}:/usr/local/Ascend/latest/lib64)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_RPATH
${ASCEND_RPATH}:/usr/local/Ascend/nnae/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64)
    set(ASCEND_RPATH
${ASCEND_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64)
    set(ASCEND_RPATH
${ASCEND_RPATH}:/usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64)
else()
    set(ASCEND_RPATH
${ASCEND_RPATH}:/usr/local/Ascend/nnae/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64)
    set(ASCEND_RPATH
${ASCEND_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64)
    set(ASCEND_RPATH
${ASCEND_RPATH}:/usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64)
endif()


### cxx api need file ###
if(ENABLE_ACL)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/ge)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)
    file(GLOB_RECURSE API_INFER_D
            "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/graph/ascend/*.cc"
            "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/model/acl/*.cc"
            "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/model/model_converter_utils/*.cc"
            "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/graph/acl/*.cc"
            "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/model/ms/*.cc"
            )
    add_library(mindspore_axx_api_d_obj OBJECT ${API_INFER_D})
    list(APPEND ASCEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:mindspore_axx_api_d_obj>)
endif()

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
target_link_libraries(mindspore_ascend PRIVATE mindspore_shared_lib)
target_link_libraries(mindspore_ascend PRIVATE proto_input mindspore::protobuf)
target_link_libraries(mindspore_ascend PRIVATE securec d_collective)
target_link_libraries(mindspore_ascend PRIVATE mindspore::ssl mindspore::crypto)
set_target_properties(mindspore_ascend PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/ascend:${ORIGIN_PATH}/../:${ASCEND_RPATH})

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(ENABLE_D)
        target_link_libraries(mindspore_ascend PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(ENABLE_D)
    find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_ascend PRIVATE ${GE_RUNNER} ${GRAPH})
    if(ENABLE_CPU)
        target_link_libraries(mindspore_ascend PRIVATE mindspore::event mindspore::event_pthreads
            mindspore::event_openssl -Wl,--no-as-needed mindspore::event_core ps_cache)
    endif()
endif()

if(MODE_ASCEND_ALL)
    MESSAGE("USE DAV LIB PATH: ${ASCEND_PATH}")
    find_library(ERROR_MANAGER error_manager ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(RUNTIME_LIB runtime ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(TSDCLIENT tsdclient HINTS ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(DATATRANSFER datatransfer HINTS ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(PROFILING msprofiler ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ACL ascendcl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ACL_DVPP acl_dvpp ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ACL_TDT_CHANNEL acl_tdt_channel ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(PLATFORM platform ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(OPT_FEATURE opt_feature ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(adump_server libadump_server.a ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
        find_library(OPTILING optiling ${ASCEND_CANN_OPP_X86_64_PATH} ${ASCEND_TOOLKIT_OPP_X86_64_PATH})
    else()
        find_library(OPTILING optiling ${ASCEND_CANN_OPP_AARCH64_PATH} ${ASCEND_TOOLKIT_OPP_AARCH64_PATH})
    endif()
    find_library(ACL_OP_COMPILER acl_op_compiler ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(CANN_KB cann_kb ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(COMPRESS compress ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(OPSKERNEL opskernel ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

    target_link_libraries(mindspore_ascend PRIVATE ${RUNTIME_LIB} ${TSDCLIENT} ${DATATRANSFER} ${ERROR_MANAGER}
            -Wl,--no-as-needed ${OPTILING} ${PLATFORM} ${ACL} ${ACL_TDT_CHANNEL} ${OPT_FEATURE} ${PROFILING}
            ${ACL_OP_COMPILER} ${CANN_KB} ${COMPRESS} ${OPSKERNEL})
    target_link_libraries(mindspore_ascend PRIVATE ${adump_server})
endif()

if(ENABLE_D)
    if(ENABLE_MPI)
        set_target_properties(_ascend_mpi PROPERTIES INSTALL_RPATH
            ${ORIGIN_PATH}:${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin/:${ASCEND_RPATH})
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernel/aicpu/aicpu_ops)
add_subdirectory(kernel/aicpu/aicpu_ops)
