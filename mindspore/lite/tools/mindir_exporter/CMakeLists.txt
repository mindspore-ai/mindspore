file(GLOB_RECURSE MINDIR_EXPORTER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${MINDIR_EXPORTER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

file(GLOB PROTO_FILE ""
        ${CORE_DIR}/proto/mind_ir.proto
        ${CCSRC_DIR}/utils/*.proto
        )
ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
add_library(mindir_proto_mid OBJECT ${PROTO_SRCS})

add_compile_definitions(MINDIR_EXPORT_TENSOR_LAYOUT_CLIP)
add_compile_definitions(COMMON_DLL)

set(MINDIR_EXPORT_DIR ${CCSRC_DIR}/transform/express_ir)
add_subdirectory(${MINDIR_EXPORT_DIR} mindir_exporter)
add_dependencies(_mindspore_transform_express_ir_obj mindir_proto_mid)

add_library(mindir_serializer_mid OBJECT
        ${MINDIR_EXPORTER_SRC_LIST}
        )
add_dependencies(mindir_serializer_mid fbs_src fbs_inner_src)
file(STRINGS "${TOP_DIR}/version.txt" VERSION)
add_definitions(-DVERSION=\"${VERSION}\")
add_library(ccsrc_debug_common_mid_ OBJECT
        ${CCSRC_DIR}/common/debug/common.cc
        ${CCSRC_DIR}/utils/compile_cache_context.cc
        ${CCSRC_DIR}/common/debug/mindir_exporter.cc
        )

target_link_libraries(mindir_serializer_mid
        _mindspore_transform_express_ir_obj
        ccsrc_debug_common_mid_
        mindir_proto_mid
        )
