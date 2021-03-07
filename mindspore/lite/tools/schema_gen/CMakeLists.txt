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
target_link_libraries(schema_gen mindspore-lite pthread)
