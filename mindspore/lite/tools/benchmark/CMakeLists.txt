# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )
add_executable(benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc
        ${COMMON_SRC})

add_dependencies(benchmark fbs_src)

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(benchmark mindspore-lite)
else()
    target_link_libraries(benchmark mindspore-lite pthread)
endif()
