# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )

add_executable(benchmark_train
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/net_train.cc
        ${COMMON_SRC})
if(WIN32)
    add_dependencies(benchmark_train fbs_src mindspore-lite_static mindspore-lite-train_static)
else()
    add_dependencies(benchmark_train fbs_src mindspore-lite_static mindspore-lite-train_static)
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    if(SUPPORT_NPU AND ANDROID_STL STREQUAL "c++_static")
        target_link_libraries(benchmark_train mindspore-lite minddata-lite mindspore-lite-train c++_shared)
    else()
        target_link_libraries(benchmark_train mindspore-lite minddata-lite mindspore-lite-train)
    endif()
else()
    if(WIN32)
        target_link_libraries(benchmark_train mindspore-lite_static mindspore-lite-train_static pthread cpu_kernel_mid
                              nnacl_mid minddata-lite)
    else()
        target_link_libraries(benchmark_train mindspore-lite mindspore-lite-train pthread minddata-lite)
    endif()
endif()
