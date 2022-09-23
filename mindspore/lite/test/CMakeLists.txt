set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TEST_DIR ${TOP_DIR}/mindspore/lite/test)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)

include_directories(${TOP_DIR})
include_directories(${TEST_DIR})
include_directories(${TOP_DIR}/mindspore/ccsrc/plugin/device/cpu/kernel)

include(${TOP_DIR}/cmake/external_libs/gtest.cmake)

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    add_definitions(-DRUNTIME_CONVERT)
endif()

file(GLOB_RECURSE TEST_UT_SRC
        ${TEST_DIR}/main.cc
        ${TEST_DIR}/common/common_test.cc
        ${TEST_DIR}/ut/src/infer_test.cc
        ${TEST_DIR}/ut/src/utils_test.cc
        ${TEST_DIR}/ut/src/scheduler_test.cc
        ${TEST_DIR}/ut/src/runtime/dynamic_mem_manager_test.cc
        ${TEST_DIR}/ut/src/registry/registry_test.cc
        ${TEST_DIR}/ut/src/registry/registry_custom_op_test.cc
        ${TEST_DIR}/st/multiple_device_test.cc
        ${TEST_DIR}/st/mindrt_parallel_runtime_test.cc
        ${TEST_DIR}/st/mix_data_type_test.cc
        ${TEST_DIR}/ut/nnacl/infer/*.cc
        ${TEST_DIR}/ut/src/runtime/kernel/arm/common/*.cc
        ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32/*.cc
        ${TEST_DIR}/ut/src/runtime/kernel/arm/string/*.cc
        ${TEST_DIR}/ut/src/api/context_c_test.cc
        ${TEST_DIR}/ut/src/api/tensor_c_test.cc
        )
if(MSLITE_ENABLE_SERVER_INFERENCE)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/api/model_parallel_runner_test.cc)
endif()

if(MSLITE_ENABLE_SERVER_INFERENCE)
    list(REMOVE_ITEM TEST_UT_SRC ${TEST_DIR}/st/mindrt_parallel_runtime_test.cc)
endif()

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/runtime/runtime_convert_tests.cc)
endif()

if(MSLITE_ENABLE_RUNTIME_PASS)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/runtime/runtime_pass_tests.cc)
endif()

if(MSLITE_ENABLE_TRAIN)
    file(GLOB_RECURSE TEST_TRAIN_UT_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32_grad/*.cc
            ${TEST_DIR}/ut/src/runtime/kernel/arm/cxx_api/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_TRAIN_UT_SRC})
endif()

if(MSLITE_ENABLE_SPARSE_COMPUTE)
    file(GLOB_RECURSE SPARSE_KERNEL_UT
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32-sparsity/*.cc
            )
    list(APPEND TEST_UT_SRC ${SPARSE_KERNEL_UT})
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    if(PLATFORM_ARM)
        file(GLOB_RECURSE TEST_GPU_UT_SRC
                ${TEST_DIR}/ut/src/runtime/kernel/opencl/*.cc
                ${TEST_DIR}/ut/src/registry/registry_gpu_custom_op_test.cc
                )
        list(APPEND TEST_UT_SRC ${TEST_GPU_UT_SRC})
    endif()
endif()

if(MSLITE_GPU_BACKEND STREQUAL cuda)
    set(CUDA_PATH $ENV{CUDA_HOME})
    include_directories(${CUDA_PATH}/include)
    include_directories(${CCSRC_DIR}/plugin/device/gpu/kernel)
    include_directories(${CUDA_PATH})
    file(GLOB_RECURSE TEST_GPU_UT_SRC
            ${TEST_DIR}/ut/src/extendrt/kernel/cuda/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_GPU_UT_SRC})
endif()

if(MSLITE_ENABLE_INT8)
    file(GLOB_RECURSE TEST_INT8_UT_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/arm/int8/*.cc
            ${TEST_DIR}/ut/nnacl/int8/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_INT8_UT_SRC})
endif()

if(MSLITE_ENABLE_FP16)
    file(GLOB_RECURSE TEST_FP16_UT_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp16/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_FP16_UT_SRC})

    if(SUPPORT_TRAIN)
        file(GLOB_RECURSE TEST_TRAIN_FP16_SRC
                ${TEST_DIR}/ut/src/runtime/kernel/arm/fp16_grad/*.cc)
        list(APPEND TEST_UT_SRC ${TEST_TRAIN_FP16_SRC})
    endif()
endif()

if(MSLITE_ENABLE_CONVERTER)
    if(MSLITE_ENABLE_TOOLS)
        set(TEST_CONVERTER_UT_SRC
                ${TEST_UT_SRC}
                ${LITE_DIR}/tools/benchmark/run_benchmark.cc
                ${LITE_DIR}/tools/benchmark/benchmark_base.cc
                ${LITE_DIR}/tools/benchmark/benchmark_unified_api.cc
                ${LITE_DIR}/tools/benchmark/benchmark_c_api.cc
                ${LITE_DIR}/tools/benchmark/benchmark.cc
                ${TEST_DIR}/st/benchmark_test.cc
                )
    endif()
    add_definitions(-DPRIMITIVE_WRITEABLE)
    file(GLOB_RECURSE TEST_CONVERTER_UT_SRC
            ${TEST_DIR}/ut/tools/converter/decomposer/svd_test.cc
            ${TEST_DIR}/ut/tools/converter/registry/*.cc
            ${TEST_DIR}/ut/tools/converter/parser/tflite/*.cc
            ${TEST_DIR}/st/converter_test.cc
            ${TEST_DIR}/st/delegate_test.cc
            ${TEST_DIR}/st/mindrt_parallel_test.cc
            ${TEST_DIR}/st/graph_test.cc
            ${TEST_DIR}/st/sub_graph_test.cc
            ${TEST_DIR}/ut/src/dynamic_library_loader_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/*.cc
            )
    if(MSLITE_ENABLE_SERVER_INFERENCE)
        list(REMOVE_ITEM TEST_CONVERTER_UT_SRC ${TEST_DIR}/st/mindrt_parallel_test.cc)
        list(REMOVE_ITEM TEST_UT_SRC ${TEST_DIR}/st/benchmark_test.cc)
        list(REMOVE_ITEM TEST_CONVERTER_UT_SRC ${TEST_DIR}/st/graph_test.cc)
        list(REMOVE_ITEM TEST_CONVERTER_UT_SRC ${TEST_DIR}/st/sub_graph_test.cc)
    endif()

    set(TEST_CONVERTER_UT_SRC
        ${TEST_CONVERTER_UT_SRC}
        ${TEST_DIR}/main.cc
        ${TEST_DIR}/common/common_test.cc
        ${TEST_DIR}/common/import_from_meta_graphT.cc
        )
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/runtime/kernel_executor_tests.cc)
endif()

set(TEST_SRC
        ${TEST_UT_SRC}
        ${TEST_LITE_SRC}
        ${TEST_DIR}/st/graph_test.cc
        )

if(MSLITE_ENABLE_SERVER_INFERENCE)
    list(REMOVE_ITEM TEST_SRC ${TEST_DIR}/st/graph_test.cc)
endif()

add_executable(lite-test ${TEST_SRC})
add_dependencies(lite-test fbs_src fbs_inner_src)

target_link_libraries(lite-test mindspore-lite dl mindspore::gtest)

# test file using gmock
target_link_libraries(lite-test mindspore::gmock)
add_library(gmock_tests OBJECT ${TEST_DIR}/st/optimize_allocator_test.cc)
add_dependencies(gmock_tests fbs_src fbs_inner_src)
target_compile_options(gmock_tests PRIVATE -fno-rtti)
target_link_libraries(lite-test gmock_tests)

if(MSLITE_ENABLE_TRAIN)
    target_link_libraries(lite-test mindspore-lite-train)
    if(NOT MSLITE_MINDDATA_IMPLEMENT STREQUAL "off")
        target_link_libraries(lite-test minddata-lite)
    endif()
endif()

if(PLATFORM_ARM)
    target_link_libraries(lite-test log)
else()
    target_link_libraries(lite-test ${SECUREC_LIBRARY} pthread)
endif()

if(MSLITE_ENABLE_NPU)
    target_link_libraries(lite-test c++_shared)
endif()

if(MSLITE_ENABLE_MINDRT)
    add_library(mindrt_test_mid OBJECT ${TEST_DIR}/ut/src/lite_mindrt_test.cc)
    add_dependencies(mindrt_test_mid fbs_src fbs_inner_src)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        target_compile_options(mindrt_test_mid PRIVATE -fno-rtti)
    endif()
    target_link_libraries(lite-test mindrt_test_mid)
endif()

if(MSLITE_ENABLE_CONVERTER AND (NOT MSLITE_ENABLE_RUNTIME_CONVERT))
    add_executable(lite-test-converter ${TEST_CONVERTER_UT_SRC})
    target_compile_definitions(lite-test-converter PRIVATE PRIMITIVE_WRITEABLE USE_GLOG)
    target_link_libraries(lite-test-converter dl mindspore::gtest)
    if(PLATFORM_ARM)
        target_link_libraries(lite-test-converter log)
    else()
        target_link_libraries(lite-test-converter ${SECUREC_LIBRARY} pthread)
    endif()
    add_dependencies(lite-test-converter fbs_src fbs_inner_src)
    target_link_libraries(lite-test-converter
            converter_runtime_mid
            cpu_ops_mid
            nnacl_mid
            cpu_kernel_mid
            lite_exporter_mid
            mslite_converter_plugin
            graph_pass_mid
            fusion_mid
            quantizer_mid
            decomposer_mid
            proto_mid
            ccsrc_src_mid
            converter_src_mid
            mindspore::protobuf
            mindspore::json
            -Wl,--whole-archive mindspore_core -Wl,--no-whole-archive
            mindspore::glog
            mindspore::flatbuffers
            preprocess_mid
            config_parser_mid
            coder_mid
            ccsrc_debug_common_mid_
            mindir_proto_mid
            _mindspore_transform_express_ir_obj
            mindir_serializer_mid
            )
    if(SUPPORT_TRAIN)
        target_link_libraries(lite-test-converter train_cpu_kernel_mid)
    endif()
endif()

if(MSLITE_ENABLE_CONVERTER)
    target_link_libraries(lite-test-converter tflite_parser_mid caffe_parser_mid
                                    onnx_parser_mid tf_parser_mid)
endif()

if(MSLITE_ENABLE_MODEL_OBF)
    target_link_libraries(lite-test ${OBF_LIB_DIR}/libmsdeobfuscator-lite.so)
endif()

if(MSLITE_ENABLE_CONVERTER AND MSLITE_ENABLE_ACL)
    target_link_libraries(lite-test lite_acl_mid mindspore_shared_lib)
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    target_link_libraries(lite-test kernel_executor)
endif()
