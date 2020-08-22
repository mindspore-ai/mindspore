project(nnacl)

set(NNACL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SRC_DIR ${TOP_DIR}/src)
set(LITE_SRC
        ${SRC_DIR}/common/graph_util.cc
        ${SRC_DIR}/common/ms_tensor_utils.cc
        ${SRC_DIR}/runtime/allocator.cc
        ${SRC_DIR}/runtime/runtime_api.cc
        ${SRC_DIR}/runtime/thread_pool.cc
        ${SRC_DIR}/runtime/workspace_pool.cc
        ${SRC_DIR}/ir/tensor.cc
        ${SRC_DIR}/context.cc
        ${SRC_DIR}/executor.cc
        ${SRC_DIR}/kernel_registry.cc
        ${SRC_DIR}/lite_kernel.cc
        ${SRC_DIR}/populate_parameter.cc
        ${SRC_DIR}/scheduler.cc
        ${SRC_DIR}/lite_session.cc
        ${SRC_DIR}/model.cc
        )

if (SUPPORT_GPU)
    set(LITE_SRC
            ${LITE_SRC}
            ${SRC_DIR}/runtime/kernel/opencl/subgraph_opencl_kernel.cc
            ${SRC_DIR}/runtime/kernel/opencl/utils.cc
            ${SRC_DIR}/runtime/opencl/opencl_executor.cc
            ${SRC_DIR}/runtime/opencl/opencl_allocator.cc
            ${SRC_DIR}/runtime/opencl/opencl_runtime.cc
            ${SRC_DIR}/runtime/opencl/opencl_wrapper.cc
            )
endif ()

set(ANF_SRC
        ${ANF_SRC}
        ${SRC_DIR}/ir/meta_tensor_extends.cc
        )
file(GLOB_RECURSE C_OPS_SRC ${SRC_DIR}/ops/*.cc)
include_directories(NNACL_DIR)

########################### optimized files ###########################
file(GLOB OPTIMIZED_ASSEMBLY
        ${NNACL_DIR}/assembly/opt/*.s
        ${NNACL_DIR}/assembly/opt/*.S
        )

file(GLOB FP16_SRC
        ${NNACL_DIR}/fp16/*.c
        ${TOP_DIR}/src/runtime/kernel/arm/fp16/*.cc
        )

########################### share library build ########################
set(OPTIMIZED_OPS ${NNACL_DIR}/opt_op_handler.c)

set_property(SOURCE ${OPTIMIZED_ASSEMBLY} PROPERTY LANGUAGE C)
list(APPEND OPTIMIZED_OPS ${OPTIMIZED_ASSEMBLY} ${FP16_SRC})

if (PLATFORM_ARM64)
    string(REPLACE "-fvisibility=hidden" "-fvisibility=default" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+fp16")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16")
    add_library(optimize SHARED ${OPTIMIZED_OPS} ${LITE_SRC} ${ANF_SRC} ${C_OPS_SRC})
    add_library(cpu_kernel STATIC $<TARGET_OBJECTS:cpu_kernel_mid_>)
    target_link_libraries(optimize cpu_kernel)
    find_library(log-lib log)
    target_link_libraries(optimize ${log-lib})
    set_target_properties(optimize PROPERTIES CLEAN_DIRECT_OUTPUT 1)

    add_custom_command(TARGET optimize POST_BUILD
            COMMAND ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
            ${TOP_DIR}/build/nnacl/liboptimize.so)

    add_custom_command(TARGET optimize POST_BUILD
            COMMAND rm -rf ${TOP_DIR}/output/lib/liboptimize.so
            COMMAND mkdir -pv ${TOP_DIR}/output/lib
            COMMAND cp ${TOP_DIR}/build/nnacl/liboptimize.so ${TOP_DIR}/output/lib)
endif ()
