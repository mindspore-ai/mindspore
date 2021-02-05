set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections ")
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(3RD_DIR ${TOP_DIR}/third_party)
set(NNACL_DIR ${LITE_DIR}/nnacl)

#include 3rd
include_directories(${3RD_DIR})
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/securec/include/)
#include ms
include_directories(${TOP_DIR}/)
include_directories(${LITE_DIR})
include_directories(${TOP_DIR}/mindspore/core/)

#include coder
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)


file(GLOB_RECURSE PRIMITIVE_OP_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITE_DIR}/src/ops/*.cc
        )

set(LITE_SRC
        ${PRIMITIVE_OP_SRC}
        ${LITE_DIR}/tools/common/flag_parser.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/string_util.cc
        ${LITE_DIR}/src/runtime/allocator.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/common/log_adapter.cc
        ${NNACL_DIR}/int8/quantize.c
        ${NNACL_DIR}/int8/pack_int8.c
        ${NNACL_DIR}/int8/matmul_int8.c
        ${NNACL_DIR}/int8/fixed_point.c
        )

file(GLOB_RECURSE MICRO_ALLOCATOR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        allocator/*.cc
        )

file(GLOB_RECURSE MICRO_GENERATOR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        generator/*.cc
        )

file(GLOB_RECURSE MICRO_OPCODERS_BASE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/base/*.cc
        )

file(GLOB_RECURSE MICRO_OPCODERS_CMSIS_NN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/cmsis-nn/*.cc
        )

file(GLOB_RECURSE MICRO_UTILS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        utils/*.cc
        )

set(MICRO_CODER_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/coder_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session_coder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/debug.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coder_graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/op_coder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/op_coder_register.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/op_coder_builder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/file_collector.cc
        )

list(APPEND MICRO_CODER_SRC
        ${MICRO_ALLOCATOR}
        ${MICRO_GENERATOR}
        ${MICRO_OPCODERS_BASE}
        ${MICRO_OPCODERS_CMSIS_NN}
        )

add_executable(codegen main.cc
        ${MICRO_UTILS}
        ${LITE_SRC}
        ${MICRO_CODER_SRC}
        ${MICRO_RUNTIME_SRC})

add_dependencies(codegen fbs_src)
add_dependencies(codegen fbs_inner_src)
target_link_libraries(codegen ${SECUREC_LIBRARY})
