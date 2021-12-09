# add shared link library
include_directories(${CCSRC_DIR}/backend/kernel_compiler/cpu)
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ccsrc/backend/kernel_compiler/cpu/nnacl/nnacl_common.c
        )
if(NOT MSLITE_ENABLE_DPICO_ACL_ADAPTER)
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

    if((PLATFORM_ARM32 OR PLATFORM_ARM64) AND NOT TARGET_HIMIX
            AND NOT TARGET_OHOS_LITE AND NOT MACHINE_LINUX_ARM64 AND NOT TARGET_MIX210)
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
else()
    __download_pkg(34xx_sdk
            http://mindspore-repo.csi.rnd.huawei.com/mindspore/enterprise/dpico/34xx_sdk.tar.gz
            f64a9129615b3b41b63debe17c6785af)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../lite)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../core)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpico/third_party/runtime)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpico/third_party/runtime/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpico/third_party/runtime/include/third_party)

    include_directories(${34xx_sdk_SOURCE_DIR}/include)
    include_directories(${34xx_sdk_SOURCE_DIR})
    link_directories(${34xx_sdk_SOURCE_DIR}/lib)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpico/third_party/runtime/lib)
    set(CMAKE_SKIP_BUILD_RPATH on)

    add_executable(benchmark
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_base.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_unified_api.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_c_api.cc
            ${COMMON_SRC})

    add_dependencies(benchmark fbs_src)

    target_link_libraries(benchmark mindspore-lite mindspore::json pthread
            dpico_acl_adapter dl svp_acl securec protobuf-c stdc++)
endif()