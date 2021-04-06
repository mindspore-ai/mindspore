set(_DEBUG_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/draw.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dump_proto.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/trace.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/env_config_parser.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_json_parser.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/cpu_e2e_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_utils.cc"
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
        )
endif()
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND _DEBUG_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/common.cc")
    list(APPEND _DEBUG_SRC_LIST "data_dump/e2e_dump.cc")
endif()

set_property(SOURCE ${_DEBUG_SRC_LIST} ${_RDR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST} ${_RDR_SRC_LIST})
