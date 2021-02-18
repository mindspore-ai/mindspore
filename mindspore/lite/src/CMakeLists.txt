add_compile_definitions(USE_ANDROID_LOG)
if(ENABLE_V0)
    add_definitions(-DENABLE_V0)
endif()
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${LITE_DIR}/nnacl/)
include_directories(${LITE_DIR}/nnacl/optimize)

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    #for performance
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
    endif()
endif()

set(API_SRC
        ${CORE_DIR}/utils/status.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/cell.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/serialization.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/types.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/lite_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/model/model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/model/model_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/graph/graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/tensor/tensor_impl.cc
)

set(LITE_SRC
        ${API_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/log_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_api.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/thread_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorlist.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/inner_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_registry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sub_graph_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/errorcode.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dequant.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/huffman_decode.cc
        )

if(SUPPORT_GPU STREQUAL opencl)
    file(GLOB_RECURSE OPENCL_RUNTIME_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/gpu/opencl/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()

if(SUPPORT_GPU STREQUAL vulkan)
    file(GLOB VULKAN_RUNTIME_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/gpu/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/gpu/vulkan/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${VULKAN_RUNTIME_SRC}
            )
endif()

if(SUPPORT_GPU STREQUAL cuda)
    file(GLOB CUDA_RUNTIME_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/gpu/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/gpu/cuda/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${CUDA_RUNTIME_SRC}
            )
endif()
if(SUPPORT_TRAIN)
    set(ANF_SRC
            ${ANF_SRC}
            )
    set(PASS_SRC)
    set(LITE_SRC
            ${LITE_SRC}
            ${ANF_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_populate_parameter.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_session.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/transfer_session.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_model.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_loop.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/train_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/loss_monitor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/lr_scheduler.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/accuracy_metrics.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/accuracy_monitor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/train/classification_train_accuracy_monitor.cc
            )
endif()

add_subdirectory(ops)
add_subdirectory(runtime/kernel/arm)

add_library(mindspore-lite SHARED ${LITE_SRC})
add_dependencies(mindspore-lite fbs_src)
set_target_properties(mindspore-lite PROPERTIES CLEAN_DIRECT_OUTPUT 1)

add_library(mindspore-lite_static STATIC ${LITE_SRC})
add_dependencies(mindspore-lite_static fbs_src)
set_target_properties(mindspore-lite_static PROPERTIES OUTPUT_NAME "mindspore-lite")
set_target_properties(mindspore-lite_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
if(SUPPORT_GPU STREQUAL opencl)
    add_subdirectory(runtime/kernel/opencl)
    target_link_libraries(mindspore-lite cpu_kernel_mid opencl_kernel_mid nnacl cpu_ops_mid)
    target_link_libraries(mindspore-lite_static cpu_kernel_mid opencl_kernel_mid nnacl_mid cpu_ops_mid)
elseif(SUPPORT_GPU STREQUAL cuda)
        add_subdirectory(runtime/kernel/cuda)
        target_link_libraries(mindspore-lite cpu_kernel_mid cuda_kernel_mid nnacl cpu_ops_mid)
        target_link_libraries(mindspore-lite_static cpu_kernel_mid cuda_kernel_mid nnacl_mid cpu_ops_mid)
else()
    target_link_libraries(mindspore-lite cpu_kernel_mid nnacl cpu_ops_mid)
    target_link_libraries(mindspore-lite_static cpu_kernel_mid nnacl_mid cpu_ops_mid)
endif()
if(SUPPORT_NPU)
    add_subdirectory(runtime/agent/npu)
    include_directories(${DDK_PATH})
    target_link_libraries(mindspore-lite npu_kernel_mid)
    target_link_libraries(mindspore-lite_static npu_kernel_mid)
endif()
if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(mindspore-lite log)
    target_link_libraries(mindspore-lite_static log)
endif()
if(BUILD_MINDDATA STREQUAL "lite")
    target_link_libraries(mindspore-lite minddata_eager_mid minddata-lite)
    target_link_libraries(mindspore-lite_static minddata_eager_mid)
endif()
if(SUPPORT_TRAIN)
    target_link_libraries(mindspore-lite minddata-lite)
endif()


if(PLATFORM_ARM)
    set(NDK_STRIP
            "${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND PLATFORM_ARM)
    add_custom_command(TARGET mindspore-lite POST_BUILD COMMAND ${NDK_STRIP}
            ${CMAKE_BINARY_DIR}/src/libmindspore-lite.so)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    if(PLATFORM_ARM)
        add_custom_command(TARGET mindspore-lite POST_BUILD COMMAND ${NDK_STRIP}
                ${CMAKE_BINARY_DIR}/src/libmindspore-lite.so)
    elseif(NOT WIN32)
        add_custom_command(TARGET mindspore-lite POST_BUILD COMMAND strip ${CMAKE_BINARY_DIR}/src/libmindspore-lite.so)
    endif()
endif()

########################## build optimize and float16 library #################################
if(PLATFORM_ARM64)
    target_link_libraries(mindspore-lite cpu_opt_kernel_mid nnacl_optimize_mid)
    target_link_libraries(mindspore-lite_static cpu_opt_kernel_mid nnacl_optimize_mid)
    if(ENABLE_FP16)
        target_link_libraries(mindspore-lite cpu_fp16_kernel_mid nnacl_fp16_mid)
        target_link_libraries(mindspore-lite_static cpu_fp16_kernel_mid nnacl_fp16_mid)
    endif()
endif()
