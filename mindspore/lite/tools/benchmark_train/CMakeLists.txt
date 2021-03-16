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
    add_dependencies(benchmark_train fbs_src mindspore-lite_static)
else()
    add_dependencies(benchmark_train fbs_src)
endif()

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(benchmark_train mindspore-lite minddata-lite)
else()
    if(WIN32)
        target_link_libraries(benchmark_train mindspore-lite_static pthread cpu_kernel_mid nnacl_mid minddata-lite)
    else()
        target_link_libraries(benchmark_train mindspore-lite pthread minddata-lite)
    endif()
endif()
