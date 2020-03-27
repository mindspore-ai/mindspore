
set(_DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/anf_ir_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/draw.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/dump_proto.cc")

if(ENABLE_DUMP_E2E)
    list(APPEND _DEBUG_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/e2e_dump.cc")
endif(ENABLE_DUMP_E2E)

add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST})