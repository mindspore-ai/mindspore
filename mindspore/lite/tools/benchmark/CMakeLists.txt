cmake_minimum_required(VERSION 3.12)
project(Lite_benchmark)

if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    set(BENCHMARK_LINK_LIB mindspore-lite)
else()
    add_definitions(-DUSE_GLOG)
    set(BENCHMARK_LINK_LIB mindspore-extendrt)
endif()

set(PROVIDERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../providers)
if(TARGET_HIMIX)
    add_subdirectory(${PROVIDERS_DIR}/nnie nnie)
    add_subdirectory(${PROVIDERS_DIR}/nnie_proposal nnie_proposal)
    set(CMAKE_SKIP_BUILD_RPATH on)
    set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} pthread
            mslite_proposal mslite_nnie dl nnie mpi  VoiceEngine upvqe dnvqe securec)
    if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3516D")
        include_directories(${PROVIDERS_DIR}/nnie/third_patry/hi3516_sdk)
        link_directories(${PROVIDERS_DIR}/nnie/third_patry/hi3516_sdk/lib)
        set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} mindspore::json)
    elseif(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3519A")
        include_directories(${PROVIDERS_DIR}/nnie/third_patry/hi3519_sdk)
        link_directories(${PROVIDERS_DIR}/nnie/third_patry/hi3519_sdk/lib)
        set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} mindspore::json)
    elseif(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3559A")
        include_directories(${PROVIDERS_DIR}/nnie/third_patry/hi3559_sdk)
        link_directories(${PROVIDERS_DIR}/nnie/third_patry/hi3559_sdk/lib)
        add_compile_definitions(BENCHMARK_CLIP_JSON)
    endif()
elseif(TARGET_MIX210 AND (NOT MSLITE_ENABLE_ACL))
    set(CMAKE_SKIP_BUILD_RPATH on)
    set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} mindspore::json pthread
        dpico_acl_adapter stdc++)
    if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "SD3403")
        add_subdirectory(${PROVIDERS_DIR}/dpico dpico)
        message("SD3403_SDK_PATH:${SD3403_SDK_PATH}.")
        include_directories(${SD3403_SDK_PATH}/include)
        include_directories(${SD3403_SDK_PATH})
        link_directories(${SD3403_SDK_PATH}/lib)
    endif()
else()
    set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} mindspore::json)
    if(PLATFORM_ARM32 OR PLATFORM_ARM64 AND NOT TARGET_OHOS_LITE AND NOT MACHINE_LINUX_ARM64)
        if(SUPPORT_NPU AND ANDROID_STL STREQUAL "c++_static")
            set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} c++_shared)
        endif()
    elseif(NOT MSVC)
        set(BENCHMARK_LINK_LIB ${BENCHMARK_LINK_LIB} pthread)
    endif()
endif()

include_directories(${CCSRC_DIR}/plugin/device/cpu/kernel)
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/opengl_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/config_file.cc
        )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../lite)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../core)

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED)
        list(APPEND opengl_lib EGL GLESv3)
        set(BENCHMARK_LINK_LIB  ${BENCHMARK_LINK_LIB} ${opengl_lib})
    endif()
endif()

if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
  set(C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_c_api.cc)
endif()

add_executable(benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_unified_api.cc
        ${C_SRC}
        ${COMMON_SRC})

add_dependencies(benchmark fbs_src)

target_link_libraries(benchmark ${BENCHMARK_LINK_LIB})
