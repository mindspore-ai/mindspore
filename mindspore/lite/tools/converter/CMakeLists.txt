add_definitions(-DPRIMITIVE_WRITEABLE)
add_definitions(-DUSE_GLOG)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

set(CCSRC_SRC
        ${CCSRC_DIR}/backend/optimizer/common/pattern_engine.cc
        ${CCSRC_DIR}/backend/optimizer/common/visit.cc
        ${CCSRC_DIR}/backend/optimizer/common/optimizer.cc
        )

include_directories(${TOP_DIR}/mindspore/ccsrc/backend/kernel_compiler/cpu)

if(NOT WIN32 AND NOT MSLITE_ENABLE_ACL)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -rdynamic -fvisibility=hidden")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -rdynamic -fvisibility=hidden")
endif()

file(GLOB_RECURSE CONVERTER_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_flags.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/anf_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graphdef_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/export_model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/meta_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/node_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/tensor_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/protobuf_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/func_graph_subgraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/storage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/meta_graph_serializer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/parse_config_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/parser_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tf_bidirection_gru_cf_fusion.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unused_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/conv1d_inout_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/inputs_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unify_format.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/cast_op_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/mindspore_importer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/primitive_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/mindir_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/mindir_control_flow_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter/acl/acl_pass.cc

        ${SRC_DIR}/common/quant_utils.cc
        ${SRC_DIR}/common/dynamic_library_loader.cc
        ${SRC_DIR}/train/train_populate_parameter.cc

        ../optimizer/*.cc
        )

if((NOT WIN32) AND MSLITE_ENABLE_DPICO_ATC_ADAPTER)
    add_subdirectory(adapter/dpico)
endif()
add_subdirectory(../anf_exporter anf_exporter)
add_subdirectory(parser/caffe)
add_subdirectory(parser/tflite)
add_subdirectory(parser/onnx)
add_subdirectory(parser/tf)
add_subdirectory(legacy_optimizer)
add_subdirectory(quantizer)
add_subdirectory(registry)
add_subdirectory(preprocess)
add_subdirectory(config_parser)
add_subdirectory(${CORE_DIR} mindspore_core)

if(MSLITE_ENABLE_ACL)
    set(MODE_ASCEND_ACL ON)
    include(${TOP_DIR}/cmake/dependency_graphengine.cmake)
    add_subdirectory(adapter/acl)
    link_directories(${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
endif()

set(API_SRC ${SRC_DIR}/cxx_api/context.cc)
if(MSLITE_ENABLE_ACL)
    list(APPEND API_SRC ${SRC_DIR}/cxx_api/kernel.cc)
endif()
set(LITE_SRC ${API_SRC}
        ${SRC_DIR}/ops/ops_def.cc
        ${SRC_DIR}/ops/ops_utils.cc
        ${SRC_DIR}/common/utils.cc
        ${SRC_DIR}/common/file_utils.cc
        ${SRC_DIR}/common/context_util.cc
        ${SRC_DIR}/common/graph_util.cc
        ${SRC_DIR}/common/string_util.cc
        ${SRC_DIR}/common/lite_utils.cc
        ${SRC_DIR}/common/log.cc
        ${SRC_DIR}/common/prim_util.cc
        ${SRC_DIR}/common/tensor_util.cc
        ${SRC_DIR}/runtime/allocator.cc
        ${SRC_DIR}/runtime/inner_allocator.cc
        ${SRC_DIR}/runtime/runtime_allocator.cc
        ${SRC_DIR}/runtime/dynamic_mem_allocator.cc
        ${SRC_DIR}/runtime/dynamic_mem_manager.cc
        ${SRC_DIR}/runtime/infer_manager.cc
        ${SRC_DIR}/runtime/runtime_pass.cc
        ${SRC_DIR}/inner_context.cc
        ${SRC_DIR}/tensor.cc
        ${SRC_DIR}/tensor_category.cc
        ${SRC_DIR}/schema_tensor_wrapper.cc
        ${SRC_DIR}/ms_tensor.cc
        ${SRC_DIR}/tensorlist.cc
        ${SRC_DIR}/kernel_registry.cc
        ${SRC_DIR}/registry/register_kernel_impl.cc
        ${SRC_DIR}/inner_kernel.cc
        ${SRC_DIR}/lite_kernel.cc
        ${SRC_DIR}/lite_kernel_util.cc
        ${SRC_DIR}/scheduler.cc
        ${SRC_DIR}/sub_graph_kernel.cc
        ${SRC_DIR}/sub_graph_split.cc
        ${SRC_DIR}/lite_session.cc
        ${SRC_DIR}/executor.cc
        ${SRC_DIR}/lite_model.cc
        ${SRC_DIR}/errorcode.cc
        ${SRC_DIR}/weight_decoder.cc
        ${SRC_DIR}/huffman_decode.cc
        ${SRC_DIR}/pack_weight_manager.cc
        ${SRC_DIR}/delegate/tensorrt/distribution/distribution_base.cc
        )
if(MSLITE_ENABLE_SERVER_INFERENCE)
set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/runtime/dynamic_mem_allocator.cc
        ${SRC_DIR}/runtime/dynamic_mem_manager.cc
        )
endif()

if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB CONTROL_FLOW_SRC
            ${SRC_DIR}/control_flow/*.cc
            )
    set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_SRC})
endif()

if(MSLITE_ENABLE_MINDRT)
    add_compile_definitions(ENABLE_MINDRT)
    include_directories(${CORE_DIR}/mindrt)
    include_directories(${CORE_DIR}/mindrt/include)
    include_directories(${CORE_DIR}/mindrt/src/)
    file(GLOB MINDRT_SRC
            ${CORE_DIR}/mindrt/src/*.cc
            ${CORE_DIR}/mindrt/src/actor/*.cc
            ${CORE_DIR}/mindrt/src/async/*.cc
            ${CORE_DIR}/mindrt/src/evloop/*.cc
            ${CORE_DIR}/mindrt/src/timer/*.cc
            ${CORE_DIR}/mindrt/src/thread/*.cc
            )

    set(LITE_SRC
            ${LITE_SRC}
            ${MINDRT_SRC}
            ${SRC_DIR}/lite_mindrt.cc
            ${SRC_DIR}/mindrt_executor.cc
            )

    if(MSLITE_ENABLE_CONTROLFLOW)
        file(GLOB CONTROL_FLOW_ACTOR_SRC
                ${SRC_DIR}/control_flow/actor/*.cc
                )
        set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_ACTOR_SRC})
    endif()
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB_RECURSE OPENCL_RUNTIME_SRC
            ${SRC_DIR}/runtime/gpu/opencl/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()

file(GLOB PROTO_FILE ""
        ${TOP_DIR}/third_party/proto/caffe/caffe.proto
        ${TOP_DIR}/third_party/proto/tensorflow/*.proto
        ${TOP_DIR}/third_party/proto/onnx/onnx.proto)

ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_library(proto_mid OBJECT ${PROTO_SRCS})
target_compile_options(proto_mid PRIVATE "-Wno-array-bounds")

set(TFLITE_FBS_FILES
        ${TOP_DIR}/third_party/proto/tensorflow/lite/schema.fbs
        )
ms_build_flatbuffers_lite(TFLITE_FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/ tflite_fbs_src
        ${CMAKE_BINARY_DIR}/schema "inner")

set_property(SOURCE ${CONVERTER_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${CCSRC_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${LITE_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(converter_src_mid OBJECT ${CONVERTER_SRC})
add_dependencies(converter_src_mid fbs_src fbs_inner_src)

add_library(ccsrc_src_mid OBJECT ${CCSRC_SRC})
add_dependencies(ccsrc_src_mid fbs_src fbs_inner_src)

add_executable(converter_lite
        main.cc
        ${LITE_SRC}
        )
add_dependencies(converter_lite fbs_src fbs_inner_src)
if((NOT WIN32) AND MSLITE_ENABLE_DPICO_ATC_ADAPTER)
    add_dependencies(converter_lite dpico_atc_adapter)
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    include_directories(${SRC_DIR}/runtime/kernel/opencl)
    target_link_libraries(converter_lite PRIVATE opencl_kernel_mid)
endif()
target_link_libraries(converter_lite PRIVATE
        ccsrc_src_mid
        converter_src_mid
        cpu_ops_mid
        nnacl_mid
        cpu_kernel_mid
        tflite_parser_mid
        tf_parser_mid
        caffe_parser_mid
        onnx_parser_mid
        anf_exporter_mid
        graph_pass_mid
        fusion_mid
        quantizer_mid
        proto_mid
        ${SECUREC_LIBRARY}
        mindspore::json
        -Wl,--whole-archive mindspore_core -Wl,--no-whole-archive
        mslite_converter_plugin
        mindspore::glog
        mindspore::protobuf
        preprocess_mid
        config_parser_mid
        )

if(MSLITE_ENABLE_ACL)
    target_link_libraries(converter_lite PRIVATE
            lite_acl_mid
            ascend_kernel_mid
            mindspore_shared_lib)
endif()

if(NOT MSVC)
    target_link_libraries(converter_lite PRIVATE pthread)
endif()

if(NOT WIN32)
    target_link_libraries(converter_lite PRIVATE dl)
endif()

if(ENABLE_MODEL_OBF)
    target_link_libraries(converter_lite PRIVATE
            ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()
