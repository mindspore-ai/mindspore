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
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_registry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/populate_parameter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lite_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model.cc
    )

if (SUPPORT_GPU)
  set(LITE_SRC
      ${LITE_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/opencl/subgraph_opencl_kernel.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel/opencl/utils.cc
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
file(GLOB_RECURSE C_OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc)
add_library(mindspore-lite SHARED ${LITE_SRC} ${C_OPS_SRC})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
add_library(anf OBJECT ${ANF_SRC})
target_link_libraries(mindspore-lite
    anf
    cpu_kernel_mid_
    )

add_subdirectory(runtime/kernel/arm)
if (PLATFORM_ARM32 OR PLATFORM_ARM64)
  target_link_libraries(mindspore-lite log)
endif ()
if (BUILD_MINDDATA)
  target_link_libraries(mindspore-lite minddata-eager minddata-lite)
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND (PLATFORM_ARM64 OR PLATFORM_ARM32))
  add_custom_command(TARGET mindspore-lite POST_BUILD
      COMMAND ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
      ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.so)
endif ()

