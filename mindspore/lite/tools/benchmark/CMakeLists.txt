# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )
if(NOT TARGET_HIMIX200)
    add_executable(benchmark
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc
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

