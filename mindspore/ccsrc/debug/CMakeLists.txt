set(_DEBUG_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/draw.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dump_proto.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/info.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/label.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/trace_info.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/trace_base.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/trace.cc"
)

if (ENABLE_DEBUGGER)
    list(APPEND _DEBUG_SRC_LIST 
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/grpc_client.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
        )
endif (ENABLE_DEBUGGER)

if (ENABLE_DUMP_E2E)
    list(APPEND _DEBUG_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/e2e_dump.cc")
endif (ENABLE_DUMP_E2E)

set_property(SOURCE ${_DEBUG_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST})
