set(_DEBUG_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/draw.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dump_proto.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/trace.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/env_config_parser.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/rdr/graph_exec_order_recorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/rdr/graph_recorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/rdr/string_recorder.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/rdr/recorder_manager.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/rdr/running_data_recorder.cc"
)

if(ENABLE_DEBUGGER)
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/grpc_client.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/tensor_summary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
        )
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND _DEBUG_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/common.cc")
    list(APPEND _DEBUG_SRC_LIST "data_dump/dump_json_parser.cc")
    list(APPEND _DEBUG_SRC_LIST "data_dump/e2e_dump_util.cc")
endif()

set_property(SOURCE ${_DEBUG_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST})
