# add shared link library
include_directories(${CCSRC_DIR}/backend/kernel_compiler/cpu)

set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ccsrc/backend/kernel_compiler/cpu/nnacl/nnacl_common.c
        )

if(MSLITE_ENABLE_SHARING_MEM_WITH_OPENGL)
    set(COMMON_SRC ${COMMON_SRC} ../common/opengl_util.cc)
endif()

add_executable(benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_unified_api.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_c_api.cc
        ${COMMON_SRC})

add_dependencies(benchmark fbs_src)

if(MSLITE_ENABLE_SHARING_MEM_WITH_OPENGL)
    list(APPEND opengl_lib EGL GLESv3)
    target_link_libraries(benchmark ${opengl_lib})
endif()

if((PLATFORM_ARM32 OR PLATFORM_ARM64) AND NOT TARGET_HIMIX AND NOT TARGET_OHOS_LITE)
    if(SUPPORT_NPU AND ANDROID_STL STREQUAL "c++_static")
        target_link_libraries(benchmark mindspore-lite mindspore::json c++_shared)
    else()
        target_link_libraries(benchmark mindspore-lite mindspore::json)
    endif()
elseif(MSVC)
    target_link_libraries(benchmark mindspore-lite mindspore::json)
else()
    target_link_libraries(benchmark mindspore-lite mindspore::json pthread)
endif()
