add_definitions(-DPRIMITIVE_WRITEABLE)
if(TARGET_AOS_ARM)
    string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-Werror" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()

set(KERNEL_EXECUTOR_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_executor_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op_converter.cc
        ${TOP_DIR}/mindspore/lite/src/common/ops/ops_utils.cc
        ${TOP_DIR}/mindspore/lite/src/common/primitive_t_utils.cc
        ${TOP_DIR}/mindspore/lite/src/common/ops/ops_def.cc
        )

if(${MSLITE_ENABLE_NPU})
    if(DEFINED ENV{HWHIAI_DDK})
        message("HIAI_DDK=$ENV{HWHIAI_DDK}")
    else()
        message(FATAL_ERROR "please set HWHIAI_DDK")
    endif()
    set(DDK_PATH "$ENV{HIAI_DDK}/ddk/ai_ddk_lib")
    set(DDK_INCLUDE_PATH "$ENV{HIAI_DDK}/ddk/ai_ddk_lib/include")
    if(PLATFORM_ARM64)
        set(DDK_LIB_PATH ${DDK_PATH}/lib64)
        set(CMAKE_CXX_FLAGS -DANDROID_ABI=arm64-v8a)
    elseif(PLATFORM_ARM32)
        set(DDK_LIB_PATH ${DDK_PATH}/lib)
        set(CMAKE_CXX_FLAGS -DANDROID_ABI=armeabi-v7a)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    include_directories(${DDK_INCLUDE_PATH})
    include_directories(${TOP_DIR}/mindspore/lite)
    link_directories(${DDK_LIB_PATH})
    set(CUSTOM_OM_CXX
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_om_kernel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_om_infer.cc
            )
    list(APPEND KERNEL_EXECUTOR_SRC ${CUSTOM_OM_CXX})
endif()

add_library(kernel_executor SHARED ${KERNEL_EXECUTOR_SRC})

add_dependencies(kernel_executor fbs_inner_src fbs_src mindspore_core)

target_link_libraries(kernel_executor
        mindspore-lite
        mindspore_core
        mindspore::json
        mindspore::flatbuffers
        mindspore::glog)
