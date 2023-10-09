string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    if(PLATFORM_ARM64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    endif()
    set(ENABLE_CPU on)
    set(LOAD_PLUGIN_STATIC on)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
    add_compile_definitions(ENABLE_CLOUD_INFERENCE)
    remove_definitions(-DBUILD_LITE_INFERENCE)

    set(MSLITE_KERNEL_PLUGIN
        ${MSLITE_KERNEL_PLUGIN}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ascend/plugin/ascend_kernel_plugin.cc)

    set(MSLITE_EXTEND_NNACL_KERNEL_LIB_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/nnacl/nnacl_lib.cc
        )

    set(MSLITE_EXTEND_DEFAULT_KERNEL_LIB_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel/default/cnode_infer_manager.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel/default/kernel_mod_kernel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel/default/default_kernel_lib.cc)

    file(GLOB DELEGATE_OPS ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ops/*.cc)

    set(MSLITE_EXTEND_RUNTIME_SRC
        ${MSLITE_KERNEL_PLUGIN}
        ${MSLITE_EXTEND_NNACL_KERNEL_LIB_SRC}
        ${MSLITE_EXTEND_DEFAULT_KERNEL_LIB_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/kernel_selector/kernel_selector.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/numa_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cpu/less_test_kernel_mod.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cpu/transpose_kernel_mod.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/base_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/single_op_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/memory_offload_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/delegate_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/default_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/memory_offload/infer_strategy_builder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_device_address.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/kernel_build_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/tensor_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/runtime_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/serialization.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/func_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/comm_group_info.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/tensorrt_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/litert_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/ascend_ge_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/ascend_native_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/tensorrt/distribution/distribution_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate_graph_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/graph_executor/litert/func_graph_reuse_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/default_graph_compiler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_runtime/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_runtime/default_graph_runtime.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_executor/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_executor/mindrt_graph_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/execution_flow.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/execution_plan.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../infer/primitive_type.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/compile_result.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/single_graph_scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/compile_result_builder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/anfnode_tensor_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_compiler/infershape_helper.cc
        ${DELEGATE_OPS}
        )

    if(MSLITE_ENABLE_BFC_MEMORY)
        set(MSLITE_EXTEND_RUNTIME_SRC ${MSLITE_EXTEND_RUNTIME_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mem_allocator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mem_manager.cc
            )
    endif()
    if(MSLITE_ENABLE_ACL)
        set(MSLITE_EXTEND_RUNTIME_SRC ${MSLITE_EXTEND_RUNTIME_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/../train/opt_allocator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel/base_kernel.cc
        )
        if(MSLITE_ASCEND_TARGET)
            set(ASCEND_NATIVE_PLUGIN
                ${CMAKE_CURRENT_SOURCE_DIR}/session/ascend_native_session.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ascend_native/ascend_native_composite_kernel.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ascend_native/ascend_native_copy_kernel.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ascend_native/ascend_native_registration_factory.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ascend_native/ascend_native_add_kernel.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ascend_native/ascend_native_matmul_kernel.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ascend_native/ascend_native_layernorm_kernel.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ascend_native/ascend_native_gather_kernel.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/delegate/ascend_native/ascend_native_encoder_kernel.cc
                ${CCSRC_DIR}/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc
                ${CCSRC_DIR}/plugin/device/ascend/hal/hccl_adapter/converter.cc
                ${CCSRC_DIR}/plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.cc
                ${CCSRC_DIR}/utils/config_manager.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/mock/ge_mock.cc
                ${CMAKE_CURRENT_SOURCE_DIR}/mock/transform_mock.cc
            )
                add_library(ascend_native_plugin SHARED ${ASCEND_NATIVE_PLUGIN})
                find_library(ge_graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
                target_link_libraries(ascend_native_plugin ${ge_graph})
                target_include_directories(ascend_native_plugin PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
                add_subdirectory(delegate/ascend_native)
                if(TARGET ascend_native_kernels_impl)
                    set(ASCEND_NATIVE_KERNELS_IMPL ascend_native_kernels_impl)
                endif()
                target_link_libraries(ascend_native_plugin ${ASCEND_NATIVE_KERNELS_IMPL} ascend_native_mid)
                set(ASCEND_TOOLKIT_PLUGIN_PATH ${ASCEND_TOOLKIT_RUNTIME_PATH}/plugin/opskernel)
                include_directories(${TOP_DIR}/graphengine/910/inc
                ${TOP_DIR}/graphengine/910/metadef/inc
                ${TOP_DIR}/graphengine/910/inc/framework
                ${TOP_DIR}/graphengine/910/inc
                ${TOP_DIR}/graphengine/910/base
                ${TOP_DIR}/graphengine/910/metadef/inc/external
                ${TOP_DIR}/graphengine/910/metadef
                ${TOP_DIR}/graphengine/910/third_party/fwkacllib
                ${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc
                ${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc/aoe
                ${CCSRC_DIR}/plugin/device/ascend/hal/hccl_adapter)
                add_subdirectory(${CCSRC_DIR}/plugin/device/ascend/hal/hccl_adapter/plugin build)
        endif()
    endif()

    include(${LITE_DIR}/cmake/ccsrc_extendrt.cmake)

    add_library(mindspore-extendrt SHARED ${MSLITE_EXTEND_RUNTIME_SRC} ${MSLITE_EXTEND_CPU_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt lite_src_common_mid)
    target_link_libraries(mindspore-extendrt lite_src_common_mid)
    add_dependencies(mindspore-extendrt fbs_src fbs_inner_src)
    add_dependencies(mindspore-extendrt mindspore-infer-anfalgo)
    add_dependencies(mindspore-extendrt mindspore-kernel-graph)
    add_subdirectory(cxx_api)
    add_subdirectory(delegate/graph_executor/litert)
    add_library(mindspore-extendrt_static STATIC ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt_static lite_src_common_mid)
    target_link_libraries(mindspore-extendrt_static lite_src_common_mid)
    add_dependencies(mindspore-extendrt_static fbs_src fbs_inner_src)
    add_dependencies(mindspore-extendrt_static mindspore-infer-anfalgo)
    add_dependencies(mindspore-extendrt_static mindspore-kernel-graph)

    add_subdirectory(${CCSRC_DIR}/backend/common/pass common_pass)
    add_subdirectory(${CCSRC_DIR}/backend/operator backend_operator)
    add_subdirectory(${CCSRC_DIR}/backend/common/optimizer mindspore_ccsrc_backend_cmmon_optimizer)
    target_link_libraries(mindspore-extendrt mindspore_infer_shared_lib_obj)
    target_link_libraries(mindspore-extendrt mindspore-infer-anfalgo
        mindspore-kernel-graph _mindspore_backend_common_optimizer_obj
        _mindspore_backend_common_pass_obj)
    target_link_libraries(mindspore-extendrt mindspore_core mindspore::protobuf)
    target_link_libraries(mindspore-extendrt_static mindspore_infer_shared_lib_obj)
    target_link_libraries(mindspore-extendrt_static mindspore-infer-anfalgo
            mindspore-kernel-graph _mindspore_backend_common_optimizer_obj
            _mindspore_backend_common_pass_obj _mindspore_backend_operator_obj)
    target_link_libraries(mindspore-extendrt_static mindspore_core mindspore::protobuf)
    add_dependencies(mindspore-extendrt_static msplugin-ge-litert)
    target_link_libraries(mindspore-extendrt_static msplugin-ge-litert)

    add_subdirectory(${LITE_DIR}/src/executor unified_executor)

    if(NOT PLATFORM_ARM)
        add_dependencies(mindspore-extendrt _mindspore_cpu_kernel_mod_depend_obj
                mindspore-lite-proto mindspore::dnnl)
        target_link_libraries(mindspore-extendrt _mindspore_cpu_kernel_mod_depend_obj
                mindspore-lite-proto mindspore::dnnl)
        add_dependencies(mindspore-extendrt_static _mindspore_cpu_kernel_mod_depend_obj
                mindspore-lite-proto mindspore::dnnl)
        target_link_libraries(mindspore-extendrt_static _mindspore_cpu_kernel_mod_depend_obj
                mindspore-lite-proto mindspore::dnnl)

        set(CPU_KERNEL_OBJECT_COUNT 0)
        add_subdirectory(${CCSRC_DIR}/plugin/device/cpu/kernel lite_kernel_mod)
        foreach(number RANGE 1 ${CPU_KERNEL_OBJECT_COUNT})
            target_link_libraries(mindspore-extendrt _mindspore_plugin_device_cpu_kernel_obj_${number})
            target_link_libraries(mindspore-extendrt_static _mindspore_plugin_device_cpu_kernel_obj_${number})
        endforeach()
    endif()

    if(NOT WIN32)
        target_link_libraries(mindspore-extendrt dl)
        target_link_libraries(mindspore-extendrt_static dl)
    endif()

    if(MSLITE_ENABLE_ACL)
        include_directories(${TOP_DIR}/graphengine/910/inc/external)
        add_subdirectory(kernel/ascend)
        add_subdirectory(delegate/ascend_ge)
    endif()

    if(SUPPORT_CUDA)
        set(CUDA_PATH $ENV{CUDA_HOME})
        include_directories(${CCSRC_DIR}/plugin/device/gpu/kernel)
        set(ENABLE_GPU on)
        add_definitions(-DENABLE_GPU)
        set(CUDA_VERSION 11.1)
        include_directories(${CUDA_PATH})
        include_directories(${CUDA_PATH}/include)
        find_package(CUDA)
        add_subdirectory(kernel/cuda)
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_53 --expt-relaxed-constexpr)
        add_subdirectory(${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops cuda_ops)
        target_link_libraries(mindspore-extendrt cuda_lite_kernel_mid cuda_ops)
        target_link_libraries(mindspore-extendrt_static cuda_lite_kernel_mid cuda_ops)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -lineinfo)
        endif()
    endif()

    if(SUPPORT_TENSORRT)
        add_definitions(-DSUPPORT_TENSORRT)
        add_subdirectory(delegate/tensorrt)
    endif()

    if(MSLITE_ENABLE_CONVERTER)
        add_subdirectory(convert)
        target_link_libraries(mindspore-extendrt -Wl,--no-as-needed mindspore_converter)
    endif()

    if(MSLITE_EXPORT_COMPUTE_IR)
        target_link_libraries(mindspore-extendrt mindspore_lite_drawer)
        target_link_libraries(mindspore-extendrt_static mindspore_lite_drawer)
    endif()
else()
    set(MSLITE_EXTEND_RUNTIME_SRC ${MODEL_LOADER_FRAMEWORK_SRC})
    add_library(mindspore-extendrt OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt fbs_src fbs_inner_src)
    add_library(mindspore-extendrt_static OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt_static fbs_src fbs_inner_src)
endif()

set_target_properties(mindspore-extendrt PROPERTIES OUTPUT_NAME "mindspore-lite")
