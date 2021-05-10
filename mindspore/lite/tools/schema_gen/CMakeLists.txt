# add shared link library

add_compile_definitions(GEN_SCHEMA_DEF)
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/log_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ops/ops_def.cc
        )
add_executable(schema_gen
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/schema_gen.cc
        ${COMMON_SRC})

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../schema/ops.fbs
        COMMAND ${CMAKE_BINARY_DIR}/tools/schema_gen/schema_gen --exportPath=${CMAKE_CURRENT_SOURCE_DIR}/../../schema/
        DEPENDS schema_gen
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Running Generate ops.fbs" VERBATIM)

add_custom_target(gen_ops ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../schema/ops.fbs)
