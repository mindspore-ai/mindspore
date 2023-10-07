# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )


set(TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_train.cc
    )

# add static securec link library
include(${TOP_DIR}/cmake/dependency_securec.cmake)

add_executable(benchmark_train
        ${TEST_SRC}
        ${COMMON_SRC})

if(WIN32)
    add_dependencies(benchmark_train fbs_src mindspore-lite_static mindspore-lite-train_static)
else()
    add_dependencies(benchmark_train fbs_src mindspore-lite_static mindspore-lite-train_static)
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    if(SUPPORT_NPU AND ANDROID_STL STREQUAL "c++_static")
        target_link_libraries(benchmark_train mindspore-lite mindspore-lite-train c++_shared securec)
    else()
        target_link_libraries(benchmark_train mindspore-lite mindspore-lite-train securec)
    endif()
else()
    if(WIN32)
        target_link_libraries(benchmark_train mindspore-lite_static mindspore-lite-train_static pthread cpu_kernel_mid
                              nnacl_mid train_cpu_kernel_mid securec)
    else()
        target_link_libraries(benchmark_train mindspore-lite mindspore-lite-train pthread securec)
    endif()
endif()

if(MSLITE_EXPORT_COMPUTE_IR)
    target_link_libraries(benchmark_train mindspore_lite_drawer)
endif()
