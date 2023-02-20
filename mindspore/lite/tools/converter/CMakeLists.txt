add_definitions(-DPRIMITIVE_WRITEABLE)
add_definitions(-DUSE_GLOG)
set(USE_GLOG on)
if(MSLITE_ENABLE_MODEL_ENCRYPTION)
    add_compile_definitions(ENABLE_OPENSSL)
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(CCSRC_SRC
        ${CCSRC_DIR}/backend/common/optimizer/pattern_engine.cc
        ${CCSRC_DIR}/backend/common/optimizer/visit.cc
        ${CCSRC_DIR}/backend/common/optimizer/graph_optimizer.cc
        ${CCSRC_DIR}/backend/operator/ops_backend_infer_function.cc
        ${CCSRC_DIR}/kernel/kernel_factory.cc
        )

if(NOT WIN32)
    set(CCSRC_SRC ${CCSRC_SRC}
        ${CCSRC_DIR}/utils/anfalgo.cc
        ${CCSRC_DIR}/utils/convert_utils.cc
        ${CCSRC_DIR}/utils/utils.cc
        ${CCSRC_DIR}/utils/parallel_context.cc
        )
endif()

if(MSLITE_ENABLE_GRAPH_KERNEL)
    add_compile_definitions(MSLITE_ENABLE_GRAPH_KERNEL)
    file(GLOB_RECURSE GRAPH_KERNEL_SRC
            ${TOOLS_DIR}/graph_kernel/common/*.cc
            ${TOOLS_DIR}/graph_kernel/converter/*.cc
            ${CCSRC_DIR}/common/graph_kernel/core/*.cc
            ${CCSRC_DIR}/common/graph_kernel/expanders/*.cc
            ${CCSRC_DIR}/common/graph_kernel/model/*.cc
            ${CCSRC_DIR}/common/graph_kernel/split_model/*.cc
            ${CCSRC_DIR}/common/graph_kernel/graph_kernel_flags.cc
            ${CCSRC_DIR}/kernel/akg/akg_kernel_json_generator.cc
            )
    set(CCSRC_SRC
            ${CCSRC_SRC}
            ${GRAPH_KERNEL_SRC}
            )
endif()

include_directories(${TOP_DIR}/mindspore/ccsrc/plugin/device/cpu/kernel)

file(GLOB_RECURSE CONVERTER_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_funcgraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_metagraph.cc
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/statistic_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/func_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/parser_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tf_bidirection_gru_cf_fusion.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unused_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/conv1d_inout_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/conv2d_transpose_input_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/inputs_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unify_format.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter/acl/plugin/acl_pass_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/*.cc
        ${SRC_DIR}/common/quant_utils.cc
        ${SRC_DIR}/common/dynamic_library_loader.cc
        ${SRC_DIR}/train/train_populate_parameter.cc
        ${SRC_DIR}/common/config_file.cc
        ${SRC_DIR}/common/config_infos.cc
        ${SRC_DIR}/common/primitive_t_utils.cc
        ../optimizer/*.cc
        )

if((NOT WIN32) AND MSLITE_ENABLE_DPICO_ATC_ADAPTER)
    add_subdirectory(adapter/dpico)
endif()
add_subdirectory(../lite_exporter lite_exporter)
add_subdirectory(../mindir_exporter mindir_serializer)
add_subdirectory(parser/caffe)
add_subdirectory(parser/tflite)
add_subdirectory(parser/onnx)
add_subdirectory(parser/tf)
if(ENABLE_CONVERT_PYTORCH_MODEL)
    add_subdirectory(parser/pytorch)
endif()

add_subdirectory(legacy_optimizer)
add_subdirectory(quantizer)
add_subdirectory(decomposer)
add_subdirectory(registry)
add_subdirectory(preprocess)
add_subdirectory(config_parser)
add_subdirectory(micro/coder)

add_subdirectory(${CORE_DIR} mindspore_core)

if(MSLITE_ENABLE_ACL)
    include_directories(${TOP_DIR}/graphengine/inc/external)
    set(MODE_ASCEND_ACL ON)
    include(${TOP_DIR}/cmake/dependency_graphengine.cmake)
    add_subdirectory(adapter/acl)
    link_directories(${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
endif()

file(GLOB CXX_API_SRCS
        ${SRC_DIR}/litert/cxx_api/*.cc
        ${SRC_DIR}/litert/cxx_api/model/*.cc
        ${SRC_DIR}/litert/cxx_api/graph/*.cc
        ${SRC_DIR}/litert/cxx_api/tensor/*.cc)

file(GLOB RUNTIME_PASS_SRCS
        ${SRC_DIR}/litert/pass/*.cc)

if(MSLITE_ENABLE_RUNTIME_PASS)
        file(GLOB KERNEL_ONLINE_FUSION_SRC ${SRC_DIR}/litert/pass/online_fusion/*.cc)
endif()

set(LITE_SRC ${API_SRC}
        ${CXX_API_SRCS}
        ${RUNTIME_PASS_SRCS}
        ${SRC_DIR}/common/ops/ops_def.cc
        ${SRC_DIR}/common/ops/ops_utils.cc
        ${SRC_DIR}/common/ops/anf_utils.cc
        ${SRC_DIR}/common/utils.cc
        ${SRC_DIR}/common/file_utils.cc
        ${SRC_DIR}/common/context_util.cc
        ${SRC_DIR}/common/graph_util.cc
        ${SRC_DIR}/common/string_util.cc
        ${SRC_DIR}/common/log.cc
        ${SRC_DIR}/common/prim_util.cc
        ${SRC_DIR}/common/tensor_util.cc
        ${SRC_DIR}/common/decrypt.cc
        ${SRC_DIR}/litert/allocator.cc
        ${SRC_DIR}/litert/inner_allocator.cc
        ${SRC_DIR}/litert/runtime_allocator.cc
        ${SRC_DIR}/litert/infer_manager.cc
        ${SRC_DIR}/litert/runtime_shape_fusion_pass.cc
        ${SRC_DIR}/litert/runtime_pass.cc
        ${SRC_DIR}/litert/inner_context.cc
        ${SRC_DIR}/litert/thread_pool_reuse_manager.cc
        ${SRC_DIR}/tensor.cc
        ${SRC_DIR}/litert/tensor_category.cc
        ${SRC_DIR}/litert/schema_tensor_wrapper.cc
        ${SRC_DIR}/tensorlist.cc
        ${SRC_DIR}/litert/kernel_registry.cc
        ${SRC_DIR}/registry/register_kernel_impl.cc
        ${SRC_DIR}/litert/lite_kernel.cc
        ${SRC_DIR}/litert/kernel_exec.cc
        ${SRC_DIR}/litert/kernel_exec_util.cc
        ${SRC_DIR}/litert/scheduler.cc
        ${SRC_DIR}/litert/sub_graph_kernel.cc
        ${SRC_DIR}/litert/sub_graph_split.cc
        ${KERNEL_ONLINE_FUSION_SRC}
        ${SRC_DIR}/litert/lite_session.cc
        ${SRC_DIR}/litert/executor.cc
        ${SRC_DIR}/litert/lite_model.cc
        ${SRC_DIR}/litert/model_manager.cc
        ${SRC_DIR}/errorcode.cc
        ${SRC_DIR}/litert/weight_decoder.cc
        ${SRC_DIR}/litert/pack_weight_manager.cc
        ${SRC_DIR}/litert/huffman_decode.cc
        ${SRC_DIR}/extendrt/delegate/tensorrt/distribution/distribution_base.cc
        ${SRC_DIR}/extendrt/delegate/plugin/tensorrt_executor_plugin.cc
        ${LITE_DIR}/src/extendrt/mock/lite_runtime/populate/base_operator_populate_register.cc
        ${SRC_DIR}/control_flow/control_flow_scheduler.cc
        ${SRC_DIR}/control_flow/control_subgraph_creator.cc
        ${SRC_DIR}/litert/kernel/ascend/plugin/ascend_kernel_plugin.cc
)

if(NOT ANDROID_NDK_TOOLCHAIN_INCLUDED)
    set(LITE_SRC
            ${LITE_SRC}
            ${SRC_DIR}/litert/kernel/ascend/plugin/ascend_kernel_plugin.cc
            )
endif()

if(MSLITE_ENABLE_MODEL_PRE_INFERENCE)
    set(LITE_SRC
            ${LITE_SRC}
            ${SRC_DIR}/common/thread_utils.cc
            ${SRC_DIR}/common/random_data_generator.cc
            )
endif()

set(MODEL_LOADER_FRAMEWORK_SRC
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${SRC_DIR}/extendrt/mindir_loader/model_loader.cc
)

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
add_compile_definitions(ENABLE_CLOUD_INFERENCE)
add_compile_definitions(SUPPORT_TRAIN)
set(SUPPORT_TRAIN on)

# string(REPLACE "-Werror" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
# string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(MINDIR_MODEL_SRC
        ${MINDIR_MODEL_SRC}
        ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model.cc
        ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model_util.cc
        ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model_convertor.cc
        ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model_loader.cc
        ${SRC_DIR}/extendrt/mindir_loader/mindir_model/kernel_mod_util.cc
        )

set(MINDIR_KERNEL_SRC
        ${MINDIR_KERNEL_SRC}
        ${SRC_DIR}/extendrt/mindir_loader/mindir_model/inner_kernel.cc)
endif()

set(LITE_SRC
        ${LITE_SRC}
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${MINDIR_MODEL_SRC}
        ${MINDIR_KERNEL_SRC}
        )


if(MSLITE_ENABLE_BFC_MEMORY)
    set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/extendrt/dynamic_mem_allocator.cc
        ${SRC_DIR}/extendrt/dynamic_mem_manager.cc
        ${SRC_DIR}/extendrt/numa_adapter.cc
        )
endif()

if(MSLITE_ENABLE_EXPERIMENTAL_KERNEL)
    set(LITE_SRC ${LITE_SRC}
        ${SRC_DIR}/../experimental/src/exec_env_utils.cc)
endif()


if(MSLITE_ENABLE_SHARING_MODEL_WEIGHT)
    set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/litert/pack_weight.cc
        )
endif()

if(MSLITE_ENABLE_DYNAMIC_THREAD_DISTRIBUTE)
    set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/litert/thread_cost_model.cc
        )
endif()

if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB CONTROL_FLOW_KERNEL_SRC
            ${SRC_DIR}/control_flow/kernel/*.cc
            )
    set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_KERNEL_SRC})
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
            ${SRC_DIR}/litert/lite_mindrt.cc
            ${SRC_DIR}/litert/parallel_lite_actor.cc
            ${SRC_DIR}/litert/mindrt_executor.cc
            ${SRC_DIR}/control_flow/control_actor_creator.cc
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
            ${SRC_DIR}/litert/kernel/gpu/opencl/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()

file(GLOB PROTO_FILE ""
        ${TOP_DIR}/third_party/proto/tensorflow/*.proto
        ${TOP_DIR}/third_party/proto/onnx/onnx.proto)
file(GLOB CAFFE_PROTO_FILE "" ${TOP_DIR}/third_party/proto/caffe/caffe.proto)
if(MSLITE_ENABLE_ACL AND MSLITE_ENABLE_ACL_QUANT_PARAM)
    set(PROTO_PATCH_FILE ${TOP_DIR}/third_party/patch/caffe/caffe_proto_acl_quant_param.patch001)
    message("patching caffe.proto < ${PROTO_PATCH_FILE}")
    execute_process(COMMAND ${Patch_EXECUTABLE} -Np1 -i ${PROTO_PATCH_FILE} -d ${TOP_DIR}
            -o ${TOP_DIR}/mindspore/lite/build/caffe.proto
            RESULT_VARIABLE Result)
    if(NOT Result EQUAL "0")
        message(FATAL_ERROR "Failed patch: ${_LF_PATCH_FILE}")
    endif()
    file(GLOB CAFFE_PROTO_FILE "" ${TOP_DIR}/mindspore/lite/build/caffe.proto)
endif()
set(PROTO_FILE ${PROTO_FILE} ${CAFFE_PROTO_FILE})

ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_library(proto_mid OBJECT ${PROTO_SRCS})
target_compile_options(proto_mid PRIVATE "-Wno-array-bounds")
target_compile_options(core_proto_obj PRIVATE "-Wno-array-bounds")

set(TFLITE_FBS_FILES
        ${TOP_DIR}/third_party/proto/tensorflow/lite/schema.fbs
        )
ms_build_flatbuffers_lite(TFLITE_FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/ tflite_fbs_src
        ${CMAKE_BINARY_DIR}/schema "inner")

set_property(SOURCE ${CONVERTER_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${CCSRC_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${LITE_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(converter_runtime_mid OBJECT ${LITE_SRC})
add_dependencies(converter_runtime_mid fbs_src fbs_inner_src)
target_compile_options(converter_runtime_mid PRIVATE "-Wno-stringop-overflow")
add_library(mindspore_converter SHARED $<TARGET_OBJECTS:converter_runtime_mid>)

add_library(converter_src_mid OBJECT ${CONVERTER_SRC})
add_dependencies(converter_src_mid fbs_src fbs_inner_src)

add_library(ccsrc_src_mid OBJECT ${CCSRC_SRC})
add_dependencies(ccsrc_src_mid fbs_src fbs_inner_src)
target_compile_definitions(ccsrc_src_mid PRIVATE BACKEND_DLL)
target_compile_definitions(converter_src_mid PRIVATE BACKEND_DLL)
target_compile_definitions(quantizer_mid PRIVATE BACKEND_DLL)

add_dependencies(mindspore_converter nnacl_mid)

if((NOT WIN32) AND MSLITE_ENABLE_DPICO_ATC_ADAPTER)
    add_dependencies(mindspore_converter dpico_atc_adapter)
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    include_directories(${SRC_DIR}/litert/kernel/opencl)
    target_link_libraries(mindspore_converter opencl_kernel_mid)
endif()

if(MSLITE_ENABLE_FP16)
    target_link_libraries(mindspore_converter
        nnacl_fp16_mid
        )
endif()

target_link_libraries(mindspore_converter
        ccsrc_src_mid
        converter_src_mid
        cpu_ops_mid
        nnacl_mid
        cpu_kernel_mid
        cast_for_converter_mid
        tflite_parser_mid
        tf_parser_mid
        caffe_parser_mid
        onnx_parser_mid
        lite_exporter_mid
        graph_pass_mid
        fusion_mid
        quantizer_mid
        decomposer_mid
        proto_mid
        ${SECUREC_LIBRARY}
        mindspore::json
        mindspore_core
        mslite_converter_plugin
        mindspore::glog
        mindspore::protobuf
        preprocess_mid
        config_parser_mid
        coder_mid
        mindir_serializer_mid
        )

if(SUPPORT_TRAIN)
    target_link_libraries(mindspore_converter train_cpu_kernel_mid)
endif()

if(ENABLE_CONVERT_PYTORCH_MODEL)
    target_link_libraries(mindspore_converter pytorch_parser_mid)
endif()

target_link_libraries(mindspore_converter
        ccsrc_debug_common_mid_
        mindir_proto_mid
        _mindspore_transform_express_ir_obj)

if(NOT MSVC)
    target_link_libraries(mindspore_converter pthread)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore_converter dl)
endif()

if(MSLITE_ENABLE_MODEL_OBF)
    target_link_libraries(mindspore_converter
            ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()

add_subdirectory(converter_lite)
