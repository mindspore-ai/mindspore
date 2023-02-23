add_compile_definitions(USE_ANDROID_LOG)

include_directories(${CCSRC_DIR}/plugin/device/cpu/kernel)
include_directories(${CORE_DIR})

set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tools)
if(NOT MSLITE_ENABLE_RUNTIME_CONVERT)
    add_compile_definitions(BUILD_LITE_INFERENCE)
endif()
if(NOT MSLITE_ENABLE_STRING_KERNEL)
    add_compile_definitions(STRING_KERNEL_CLIP)
endif()
if(NOT MSLITE_ENABLE_CONTROLFLOW)
    add_compile_definitions(CONTROLFLOW_TENSORLIST_CLIP)
endif()
if(NOT MSLITE_ENABLE_AUTO_PARALLEL)
    add_compile_definitions(AUTO_PARALLEL_CLIP)
endif()
if(NOT MSLITE_ENABLE_WEIGHT_DECODE)
    add_compile_definitions(WEIGHT_DECODE_CLIP)
endif()
if(NOT MSLITE_ENABLE_CUSTOM_KERNEL)
    add_compile_definitions(CUSTOM_KERNEL_REGISTRY_CLIP)
endif()
if(NOT MSLITE_ENABLE_RUNTIME_PASS)
    add_compile_definitions(RUNTIME_PASS_CLIP)
endif()
if(NOT MSLITE_ENABLE_DELEGATE)
    add_compile_definitions(DELEGATE_CLIP)
endif()
if(NOT MSLITE_ENABLE_INT8)
    add_compile_definitions(OP_INT8_CLIP)
endif()
if(APPLE OR PLATFORM_ARM32 OR PLATFORM_ARM64)
    #for performance
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND NOT APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti")
        if(TARGET_HIMIX200 OR TARGET_MIX210)
            string(REPLACE "-fno-rtti " "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
        endif()
        if(NOT MACHINE_LINUX_ARM64)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
        endif()
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -Wno-shorten-64-to-32 \
        -fno-aligned-allocation -DTARGET_OS_OSX")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -Wno-shorten-64-to-32 \
        -fno-aligned-allocation -DTARGET_OS_OSX")
        if(NOT MSLITE_ENABLE_COREML)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
        endif()
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND NOT MSLITE_ENABLE_TESTCASES AND NOT MSLITE_ENABLE_KERNEL_EXECUTOR)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    endif()
elseif(NOT MSVC)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti")
    endif()
endif()

file(GLOB CXX_API_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/model/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/graph/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/tensor/*.cc
        )
if(MSLITE_ENABLE_PARALLEL_INFERENCE)
    set(CXX_API_SRCS
            ${CXX_API_SRCS}
            ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/cxx_api/model_pool/predict_task_queue.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/cxx_api/model_pool/model_worker.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/cxx_api/model_pool/model_pool.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/cxx_api/model_pool/model_parallel_runner.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/cxx_api/model_pool/model_parallel_runner_impl.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/cxx_api/model_pool/resource_manager.cc
            )
endif()

file(GLOB C_API_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/c_api/*.cc
        )
set(API_SRC ${CXX_API_SRCS} ${C_API_SRCS})
if(NOT MSLITE_ENABLE_RUNTIME_CONVERT)
    set(API_SRC ${API_SRC} ${CORE_DIR}/utils/status.cc)
    set(LITE_SRC ${LITE_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/common/config_file.cc)
endif()
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
    file(GLOB CXX_API_TRAIN_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/train/model.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/train/model_impl.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/metrics/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/callback/*.cc
            )
    set(API_TRAIN_SRC
            ${CXX_API_TRAIN_SRCS}
            )
endif()

if(SUPPORT_NPU)
    include_directories(${DDK_PATH})
    include_directories(${DDK_INCLUDE_PATH})
endif()

set(LITE_SRC
        ${LITE_SRC}
        ${API_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/context_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/mmap_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/prim_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/tensor_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/inner_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/runtime_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/infer_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/runtime_shape_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/runtime_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/schema_tensor_wrapper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorlist.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/tensor_category.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/inner_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/lite_model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/kernel_registry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/lite_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/kernel_exec.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/kernel_exec_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/sub_graph_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/lite_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/runtime_packed_node_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/model_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/errorcode.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cpu_info.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/pack_weight_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/control_flow/control_flow_scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/control_flow/control_subgraph_creator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/thread_pool_reuse_manager.cc
        )

file(GLOB FORMAT_PASS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/litert/pass/format_pass/*.cc")
set(LITE_SRC ${LITE_SRC} ${FORMAT_PASS_SRC})

if(NOT ANDROID_NDK_TOOLCHAIN_INCLUDED)
    set(LITE_SRC
            ${LITE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/common/dynamic_library_loader.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/kernel/ascend/plugin/ascend_kernel_plugin.cc
            )
endif()

if(MSLITE_ENABLE_MODEL_PRE_INFERENCE)
    set(LITE_SRC
            ${LITE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/common/thread_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/common/random_data_generator.cc
            )
endif()

set(MODEL_LOADER_FRAMEWORK_SRC
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/mindir_loader/model_loader.cc
)

set(LITE_SRC
        ${LITE_SRC}
        ${MODEL_LOADER_FRAMEWORK_SRC}
        )

if(MSLITE_ENABLE_MODEL_ENCRYPTION)
    set(LITE_SRC
            ${LITE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/common/decrypt.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/common/dynamic_library_loader.cc
            )
endif()

if(MSLITE_ENABLE_CAPTURE_SIGNALS)
    set(LITE_SRC
        ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/signal_handler.cc
        )
endif()

if(MSLITE_ENABLE_BFC_MEMORY)
    set(LITE_SRC
        ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/dynamic_mem_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/dynamic_mem_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/extendrt/numa_adapter.cc
        )
endif()

if(MSLITE_ENABLE_SHARING_MODEL_WEIGHT)
    set(LITE_SRC
        ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/pack_weight.cc
        )
endif()

if(MSLITE_ENABLE_DYNAMIC_THREAD_DISTRIBUTE)
    set(LITE_SRC
        ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/thread_cost_model.cc
        )
endif()

if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB CONTROL_FLOW_KERNEL_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/control_flow/kernel/*.cc
            )
    set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_KERNEL_SRC})
endif()

if(MSLITE_ENABLE_EXPERIMENTAL_KERNEL)
    file(GLOB EXPERIMENT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../experimental/src/exec_env_utils.cc)
    set(LITE_SRC ${LITE_SRC} ${EXPERIMENT_SRC})
endif()

if(MSLITE_ENABLE_RUNTIME_GLOG)
    set(USE_GLOG on)
    add_definitions(-DUSE_GLOG)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    if(NOT MSLITE_ENABLE_RUNTIME_CONVERT AND NOT MSLITE_ENABLE_KERNEL_EXECUTOR)
        set(LITE_SRC ${LITE_SRC}
            ${CORE_DIR}/utils/log_adapter.cc)
    endif()
endif()

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    add_definitions(-DRUNTIME_CONVERT)
    add_definitions(-DPRIMITIVE_WRITEABLE)
    file(GLOB RUNTIME_CONVERT_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/common/ops/ops_def.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/common/ops/ops_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/common/ops/anf_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/runtime_convert.cc)

    set(LITE_SRC ${LITE_SRC} ${RUNTIME_CONVERT_SRC})
endif()

if(MSLITE_ENABLE_STRING_KERNEL)
    set(LITE_SRC
        ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/string_util.cc
        )
endif()

if(MSLITE_ENABLE_WEIGHT_DECODE)
    set(LITE_SRC ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/huffman_decode.cc)
    if(NOT MSLITE_ENABLE_RUNTIME_CONVERT)
        set(LITE_SRC ${LITE_SRC}
            ${TOOLS_DIR}/converter/quantizer/fse_decoder.cc
            ${TOOLS_DIR}/converter/quantizer/fse_bit_stream.cc)
    endif()
endif()

if(MSLITE_ENABLE_RUNTIME_PASS)
    file(GLOB KERNEL_ONLINE_FUSION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/litert/pass/online_fusion/*.cc)
    set(LITE_SRC
        ${LITE_SRC}
        ${KERNEL_ONLINE_FUSION_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/sub_graph_split.cc
    )
endif()

if(MSLITE_ENABLE_AUTO_PARALLEL)
    set(LITE_SRC
        ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/sub_graph_split.cc
    )
endif()

if(MSLITE_ENABLE_CUSTOM_KERNEL)
    file(GLOB KERNEL_REG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/registry/*.cc)
endif()

set(LITE_SRC
    ${LITE_SRC}
    ${KERNEL_REG_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/litert/weight_decoder.cc
    )

if(MSLITE_GPU_BACKEND STREQUAL cuda)
    file(GLOB CUDA_RUNTIME_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/gpu/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/gpu/cuda/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${CUDA_RUNTIME_SRC}
            )
endif()
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
  set(TRAIN_SRC_WITH_MD
          ${CMAKE_CURRENT_SOURCE_DIR}/train/train_loop.cc
          )
endif()


file(GLOB CXX_API_EXPRESSION
  ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/expression/*.cc
  )

file(GLOB EXPRESSION_OPS
  ${CMAKE_CURRENT_SOURCE_DIR}/expression/ops/*.cc
  )

set(EXPRESSION_SRC
    ${CXX_API_EXPRESSION}
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/export.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/expr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/import.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/net.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/node.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/ops_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/sequential.cc
    ${EXPRESSION_OPS}
  )

set(TRAIN_SRC
        ${API_TRAIN_SRC}
        ${TRAIN_SRC_WITH_MD}
        ${EXPRESSION_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/quant_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/metrics/accuracy.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/train/model_build.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/train/model_build_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/train/converters.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/cxx_api/train/train_support.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/train_populate_parameter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/train_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/graph_fusion.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/graph_dropout.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/transfer_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/train_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/loss_monitor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/lr_scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/accuracy_metrics.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/accuracy_monitor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/classification_train_accuracy_monitor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/train_export.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/opt_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/optimizer/common/fusion_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/optimizer/fusion/matmul_activation_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/train/optimizer/fusion/reshape_gather_reshape_fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/storage.cc
        ${TOOLS_DIR}/converter/optimizer.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/fusion/fusion_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/fusion/fusion_pattern.cc
        ${TOOLS_DIR}/common/meta_graph_utils.cc
        ${TOOLS_DIR}/common/statistic_utils.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/fusion/matmul_biasadd_fusion_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/graph/dropout_node_remove_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/graph/isolated_node_remove_pass.cc
        ${TOOLS_DIR}/converter/legacy_optimizer/graph/subgraph_node_pass.cc
        )

if(MSLITE_ENABLE_KERNEL_EXECUTOR AND NOT MSLITE_ENABLE_CONVERTER)
    add_subdirectory(${CORE_DIR} mindspore_core)
endif()

if(MSLITE_ENABLE_MINDRT)
    if((NOT MSLITE_ENABLE_CONVERTER) AND (NOT MSLITE_ENABLE_KERNEL_EXECUTOR))
        add_subdirectory(${CORE_DIR}/mindrt mindspore_mindrt)
    endif()
    set(LITE_SRC ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/lite_mindrt.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/parallel_lite_actor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/mindrt_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/control_flow/control_actor_creator.cc
        )
    if(MSLITE_ENABLE_CONTROLFLOW)
        file(GLOB CONTROL_FLOW_ACTOR_SRC
                ${CMAKE_CURRENT_SOURCE_DIR}/control_flow/actor/*.cc
                )
        set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_ACTOR_SRC})
    endif()
else()
    set(LITE_SRC ${LITE_SRC}
        ${CORE_DIR}/mindrt/src/thread/core_affinity.cc
        ${CORE_DIR}/mindrt/src/thread/threadpool.cc
        )
endif()

if(MSLITE_ENABLE_GRAPH_KERNEL)
    file(GLOB_RECURSE GRAPH_KERNEL_SRC
        ${TOOLS_DIR}/graph_kernel/common/*.cc
        ${TOOLS_DIR}/graph_kernel/runtime/*.cc
        )
    set(LITE_SRC ${LITE_SRC} ${GRAPH_KERNEL_SRC})
endif()

if(NOT MSLITE_ENABLE_COREML)
    set(LITE_SRC ${LITE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/litert/delegate/coreml/stub/coreml_delegate_stub.cc)
endif()

if(MSVC)
    set(LITE_SRC ${LITE_SRC}
    ${EXPRESSION_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/storage.cc)
endif()

add_subdirectory(litert/kernel/cpu)

add_library(lite_src_mid OBJECT ${LITE_SRC})
add_dependencies(lite_src_mid fbs_src)

if(MSLITE_ENABLE_ACL)
    include_directories(${TOP_DIR}/graphengine/inc/external)
    if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        add_subdirectory(litert/kernel/ascend)
    else()
        add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
        add_compile_definitions(ENABLE_CLOUD_INFERENCE)
    endif()
    link_directories(${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
endif()

add_library(mindspore-lite SHARED $<TARGET_OBJECTS:lite_src_mid>)
set_target_properties(mindspore-lite PROPERTIES CLEAN_DIRECT_OUTPUT 1)

if(APPLE)
    file(GLOB_RECURSE MINDSPORE_LITE_PUB_HDRS LIST_DIRECTORIES true
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/api/*.h)
    file(GLOB MINDSPORE_LITE_PUB_HDRS_IR_HDRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/ir/dtype/type_id.h)
    file(GLOB MINDSPORE_LITE_PUB_HDRS_MINDAPI_HDRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/mindapi/base/format.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/mindapi/base/type_id.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core/mindapi/base/types.h
            )

    if(NOT MSLITE_ENABLE_COREML)
        add_library(mindspore-lite_static STATIC
                ${LITE_SRC}
                ${MINDSPORE_LITE_PUB_HDRS}
                ${MINDSPORE_LITE_PUB_HDRS_MINDAPI_HDRS}
                ${MINDSPORE_LITE_PUB_HDRS_IR_HDRS}
                )
    else()
        file(GLOB_RECURSE MINDSPORE_LITE_PUB_HDRS_FBS_HDRS LIST_DIRECTORIES true ${flatbuffers_INC}/flatbuffers/*)
        add_library(mindspore-lite_static STATIC
                ${LITE_SRC}
                ${MINDSPORE_LITE_PUB_HDRS}
                ${MINDSPORE_LITE_PUB_HDRS_MINDAPI_HDRS}
                ${MINDSPORE_LITE_PUB_HDRS_IR_HDRS}
                ${MINDSPORE_LITE_PUB_HDRS_FBS_HDRS}
                )
        add_subdirectory(litert/delegate/coreml)
        target_link_libraries(mindspore-lite_static coreml_proto_mid coreml_kernel_mid)
    endif()
    add_dependencies(mindspore-lite_static fbs_inner_src)
else()
    add_library(mindspore-lite_static STATIC $<TARGET_OBJECTS:lite_src_mid>)
endif()
set_target_properties(mindspore-lite_static PROPERTIES OUTPUT_NAME "mindspore-lite")
set_target_properties(mindspore-lite_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
endif()

if(MSVC)
    set_target_properties(mindspore-lite PROPERTIES PREFIX lib)
    set_target_properties(mindspore-lite PROPERTIES IMPORT_PREFIX lib)
    set_target_properties(mindspore-lite PROPERTIES IMPORT_SUFFIX .dll.lib)
    set_target_properties(mindspore-lite_static PROPERTIES PREFIX lib)
endif()

# if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE)
# target_link_libraries(mindspore-lite cpu_kernel_mid nnacl_mid cpu_ops_mid mindspore_core mindspore::protobuf)
# target_link_libraries(mindspore-lite_static cpu_kernel_mid nnacl_mid cpu_ops_mid mindspore_core mindspore::protobuf)
# else()
target_link_libraries(mindspore-lite cpu_kernel_mid nnacl_mid cpu_ops_mid)
target_link_libraries(mindspore-lite_static cpu_kernel_mid nnacl_mid cpu_ops_mid)
# endif()

# target_link_libraries(mindspore-lite mindspore-extendrt)
# target_link_libraries(mindspore-lite_static mindspore-extendrt)

if(SUPPORT_TRAIN)
    target_link_libraries(mindspore-lite train_cpu_kernel_mid)
    target_link_libraries(mindspore-lite_static train_cpu_kernel_mid)
endif()

if(MSLITE_ENABLE_MINDRT)
    target_link_libraries(mindspore-lite mindrt_mid)
    target_link_libraries(mindspore-lite_static mindrt_mid)
endif()

if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    if(SUPPORT_TENSORRT)
        add_compile_definitions(GPU_TENSORRT)
        set(TENSORRT_PATH $ENV{TENSORRT_PATH})
        set(CUDA_PATH $ENV{CUDA_HOME})
        set(TENSORRT_LIB_PATH ${TENSORRT_PATH}/lib)
        set(CUDA_LIB_PATH ${CUDA_PATH}/lib64)
        include_directories(${TENSORRT_PATH}/include)
        include_directories(${CUDA_PATH}/include)
        add_subdirectory(litert/delegate/tensorrt)
        target_link_libraries(mindspore-lite tensorrt_kernel_mid cuda_kernel_mid gpu_distribution_collective)
        target_link_libraries(mindspore-lite_static tensorrt_kernel_mid cuda_kernel_mid gpu_distribution_collective)
    else()
        set(TENSORRT_STUB
            ${CMAKE_CURRENT_SOURCE_DIR}/litert/delegate/tensorrt/distribution/distribution_base.cc
        )
        add_library(tensorrt_stub OBJECT ${TENSORRT_STUB})
        target_link_libraries(mindspore-lite tensorrt_stub)
        target_link_libraries(mindspore-lite_static tensorrt_stub)
    endif()
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    add_subdirectory(litert/kernel/gpu/opencl)
    target_link_libraries(mindspore-lite opencl_kernel_mid)
    target_link_libraries(mindspore-lite_static opencl_kernel_mid)
endif()

if(SUPPORT_NPU)
    add_subdirectory(litert/delegate/npu)
    target_link_libraries(mindspore-lite npu_kernel_mid)
    target_link_libraries(mindspore-lite_static npu_kernel_mid)
endif()

if(SUPPORT_NNAPI)
    add_compile_definitions(SUPPORT_NNAPI)
    add_subdirectory(litert/delegate/nnapi)
    target_link_libraries(mindspore-lite nnapi_kernel_mid)
    target_link_libraries(mindspore-lite_static nnapi_kernel_mid)
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64 AND NOT TARGET_HIMIX
        AND NOT TARGET_MIX210 AND NOT TARGET_OHOS_LITE AND NOT MACHINE_LINUX_ARM64 AND NOT TARGET_OHOS)
    target_link_libraries(mindspore-lite log)
    target_link_libraries(mindspore-lite_static log)
endif()

if(TARGET_OHOS)
    target_link_libraries(mindspore-lite hilog_ndk.z.so)
    target_link_libraries(mindspore-lite_static hilog_ndk.z.so)
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite")
    target_link_libraries(mindspore-lite minddata_eager_mid minddata-lite)
    target_link_libraries(mindspore-lite_static minddata_eager_mid)
endif()

if(SUPPORT_TRAIN)
  add_library(lite_train_src_mid OBJECT ${TRAIN_SRC})
  add_dependencies(lite_train_src_mid fbs_src fbs_inner_src)

  add_library(mindspore-lite-train SHARED $<TARGET_OBJECTS:lite_train_src_mid>)
  set_target_properties(mindspore-lite-train PROPERTIES OUTPUT_NAME "mindspore-lite-train")
  set_target_properties(mindspore-lite-train PROPERTIES CLEAN_DIRECT_OUTPUT 1)
  if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
      target_link_libraries(mindspore-lite-train minddata-lite mindspore-lite)
  else()
      target_link_libraries(mindspore-lite-train mindspore-lite)
  endif()

  add_library(mindspore-lite-train_static STATIC $<TARGET_OBJECTS:lite_train_src_mid>)
  set_target_properties(mindspore-lite-train_static PROPERTIES OUTPUT_NAME "mindspore-lite-train")
  set_target_properties(mindspore-lite-train_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
  if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
      target_link_libraries(mindspore-lite-train_static minddata-lite mindspore-lite)
  else()
      target_link_libraries(mindspore-lite-train_static mindspore-lite)
  endif()
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    add_subdirectory(litert/cxx_api/kernel_executor)
endif()

########################## build optimize and float16 library #################################
if(PLATFORM_ARM)
    if(PLATFORM_ARM64 AND NOT TARGET_HIMIX AND NOT TARGET_MIX210 AND NOT MACHINE_LINUX_ARM64)
        target_link_libraries(mindspore-lite cpu_opt_kernel_mid nnacl_optimize_mid)
        target_link_libraries(mindspore-lite_static cpu_opt_kernel_mid nnacl_optimize_mid)
    endif()
    if(MSLITE_ENABLE_FP16)
        target_link_libraries(mindspore-lite cpu_fp16_kernel_mid nnacl_fp16_mid)
        target_link_libraries(mindspore-lite_static cpu_fp16_kernel_mid nnacl_fp16_mid)
    endif()
endif()

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    target_link_libraries(mindspore-lite
            quantizer_mid fusion_mid proto_mid graph_pass_mid preprocess_mid
            cpu_kernel_mid ccsrc_src_mid converter_src_mid lite_exporter_mid
            config_parser_mid mslite_converter_plugin mindspore_core coder_mid
            mindir_serializer_mid mindspore::protobuf ${SECUREC_LIBRARY})
    target_link_libraries(mindspore-lite_static
            quantizer_mid fusion_mid proto_mid graph_pass_mid preprocess_mid
            cpu_kernel_mid ccsrc_src_mid converter_src_mid lite_exporter_mid
            config_parser_mid mslite_converter_plugin mindspore_core coder_mid
            mindir_serializer_mid mindspore::protobuf ${SECUREC_LIBRARY})
    target_link_libraries(mindspore-lite
            ccsrc_debug_common_mid_ mindir_proto_mid _mindspore_transform_express_ir_obj)
    target_link_libraries(mindspore-lite_static
            ccsrc_debug_common_mid_ mindir_proto_mid _mindspore_transform_express_ir_obj)
endif()

if(SUPPORT_CUDA AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    set(CUDA_PATH $ENV{CUDA_HOME})
    include_directories(${CCSRC_DIR}/plugin/device/gpu/kernel)
    set(ENABLE_GPU on)
    add_definitions(-DENABLE_GPU)
    set(CUDA_VERSION 11.1)
    include_directories(${CUDA_PATH})
    include_directories(${CUDA_PATH}/include)
    find_package(CUDA)
    add_subdirectory(extendrt/kernel/cuda)
    list(APPEND CUDA_NVCC_FLAGS -arch=sm_53 --expt-relaxed-constexpr)
    add_subdirectory(${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops cuda_ops)
    target_link_libraries(mindspore-lite cuda_lite_kernel_mid cuda_ops mindspore_core)
endif()

if(MSLITE_ENABLE_RUNTIME_GLOG)
    target_link_libraries(mindspore-lite mindspore::glog)
    target_link_libraries(mindspore-lite_static mindspore::glog)
endif()

if(DEFINED ARCHS)
    set_target_properties(mindspore-lite_static PROPERTIES FRAMEWORK TRUE)
    set_target_properties(mindspore-lite_static PROPERTIES
            OUTPUT_NAME "mindspore-lite"
            FRAMEWORK_VERSION C
            XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
            XCODE_ATTRIBUTE_INSTALL_PATH "@rpath"
            )
    FOREACH(HDR ${MINDSPORE_LITE_PUB_HDRS})
        SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/include/api/)
    ENDFOREACH()
    FOREACH(HDR ${MINDSPORE_LITE_PUB_HDRS_IR_HDRS})
        SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/include/ir/dtype/)
    ENDFOREACH()
    FOREACH(HDR ${MINDSPORE_LITE_PUB_HDRS_MINDAPI_HDRS})
        SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/include/mindapi/base/)
    ENDFOREACH()
    if(MSLITE_ENABLE_COREML)
        FOREACH(HDR ${MINDSPORE_LITE_PUB_HDRS_FBS_HDRS})
            SET_SOURCE_FILES_PROPERTIES(
                    ${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/include/third_party/flatbuffers/)
        ENDFOREACH()
    endif()
    target_link_libraries(mindspore-lite_static)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore-lite dl)
endif()

if(MSLITE_ENABLE_MODEL_OBF)
    target_link_libraries(mindspore-lite ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
    target_link_libraries(mindspore-lite_static ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    target_link_libraries(mindspore-lite mindspore_core)
    target_link_libraries(mindspore-lite_static mindspore_core)
endif()

add_subdirectory(extendrt)
