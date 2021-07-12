# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )
if(NOT TARGET_HIMIX200)
    add_executable(benchmark
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_base.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_unified_api.cc
            ${COMMON_SRC})

    add_dependencies(benchmark fbs_src)

    if(PLATFORM_ARM32 OR PLATFORM_ARM64)
        if(SUPPORT_NPU AND ANDROID_STL STREQUAL "c++_static")
            target_link_libraries(benchmark mindspore-lite mindspore::json c++_shared)
        else()
            target_link_libraries(benchmark mindspore-lite mindspore::json)
        endif()
    else()
        target_link_libraries(benchmark mindspore-lite mindspore::json pthread)
    endif()
endif()

if(MSLITE_COMPILE_NNIE AND TARGET_HIMIX200 AND PLATFORM_ARM)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nnie/third_patry/hi3516_sdk)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/nnie/third_patry/hi3516_sdk/lib)
    set(CMAKE_SKIP_BUILD_RPATH on)
    add_subdirectory(nnie)
    add_subdirectory(nnie_proposal)

    add_executable(benchmark
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_base.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_unified_api.cc
            ${COMMON_SRC})

    add_dependencies(benchmark fbs_src)

    target_link_libraries(benchmark mindspore-lite mindspore::json pthread nnie_proposal
            mslite_nnie dl nnie mpi  VoiceEngine upvqe dnvqe securec)
endif()
