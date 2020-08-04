set(LITE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ms_tensor_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_api.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/thread_pool.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/workspace_pool.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ir/tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_registry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/populate_parameter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cc
        )

if (SUPPORT_GPU)
  list(APPEND LITE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/opencl/subgraph_opencl_kernel.cc)
  list(APPEND LITE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/opencl/utils.cc)
endif()

if (SUPPORT_TRAIN)
    set(ANF_SRC
#            ${CCSRC_DIR}/common/trans.cc
#            ${CCSRC_DIR}/utils/lite/base_ref_utils.cc
#            ${CCSRC_DIR}/runtime/kernel/kernel_compiler/kernel_build_info.cc
#            ${CCSRC_DIR}/session/lite/anf_runtime_algorithm_extends.cc
#            ${CCSRC_DIR}/session/lite/session_basic_extends.cc
#            ${CCSRC_DIR}/session/anf_runtime_algorithm.cc
#            ${CCSRC_DIR}/session/session_basic.cc
#            ${CCSRC_DIR}/session/kernel_graph.cc
#            ${CCSRC_DIR}/session/session_factory.cc
#            ${CCSRC_DIR}/device/kernel_info.cc
#            ${CCSRC_DIR}/device/kernel_runtime.cc
#            ${CCSRC_DIR}/device/lite/kernel_runtime_extends.cc
            )
    set(PASS_SRC)
    set(LITE_SRC
            ${LITE_SRC}
            ${ANF_SRC}
            ${PASS_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/common/anf_importer/anf_importer.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/common/anf_importer/import_from_meta_graph.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/ir/primitive_value.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/lite_kernel_runtime.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_session.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/model_impl.cc
            )
else ()
    set(LITE_SRC
            ${LITE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/lite_session.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/model_impl.cc
            )
endif ()

if (SUPPORT_GPU)
    set(LITE_SRC
            ${LITE_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_executor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_allocator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_runtime.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/opencl/opencl_wrapper.cc
            )
endif ()

set(ANF_SRC
        ${ANF_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ir/meta_tensor_extends.cc
        )

add_library(mindspore-lite SHARED ${LITE_SRC} ${ANF_SRC})
target_link_libraries(mindspore-lite
        cpu_kernel_mid_
        ops_mid_
        ${SECUREC_LIBRARY}
        mindspore::json
        )

add_subdirectory(runtime/kernel/arm)
add_subdirectory(ops)

