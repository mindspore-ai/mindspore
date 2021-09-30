include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/debug/)
include_directories(${CMAKE_BINARY_DIR})

set(_DEBUG_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/draw.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dump_proto.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/trace.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/env_config_parser.cc"
)

set(_OFFLINE_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/debugger/tensor_summary.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/debugger/offline_debug/dbg_services.cc"
    "${CMAKE_SOURCE_DIR}/mindspore/core/utils/log_adapter.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/debugger/offline_debug/mi_pybind_register.cc"
)

if(ENABLE_DUMP_IR)
    file(GLOB_RECURSE _RDR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "rdr/*.cc")
    if(NOT ENABLE_D)
        list(REMOVE_ITEM _RDR_SRC_LIST "rdr/task_debug_info_recorder.cc")
    endif()
endif()

if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_DEBUGGER)
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/grpc_client.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/tensor_summary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger_utils.cc"
        )
endif()

if(NOT ENABLE_SECURITY)
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/cpu_e2e_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_json_parser.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/npy_header.cc"
        )
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        list(APPEND _DEBUG_SRC_LIST
            "${CMAKE_CURRENT_SOURCE_DIR}/common.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/e2e_dump.cc"
            )
    endif()
endif()

set_property(SOURCE ${_DEBUG_SRC_LIST} ${_RDR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST} ${_RDR_SRC_LIST})
if(ENABLE_DEBUGGER)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        add_compile_options(-Wall -DOFFLINE_DBG_MODE -fPIC -O2)
        set_property(SOURCE ${_OFFLINE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_OFFLINE_DEBUG)
        add_library(_mindspore_offline_debug SHARED ${_OFFLINE_SRC_LIST})
        target_link_libraries(_mindspore_offline_debug PRIVATE mindspore::glog mindspore_gvar)
        set_target_properties(_mindspore_offline_debug PROPERTIES
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
            )
    endif()
endif()
