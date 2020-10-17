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
if (WIN32)
    add_dependencies(benchmark fbs_src mindspore-lite_static)
else ()
    add_dependencies(benchmark fbs_src)
endif ()

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(benchmark mindspore-lite)
else()
    if (WIN32)
        target_link_libraries(benchmark mindspore-lite_static pthread cpu_kernel_mid nnacl_mid)
    else ()
        target_link_libraries(benchmark mindspore-lite pthread)
    endif ()
endif()
if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    install(TARGETS benchmark
                RUNTIME DESTINATION ${MAIN_DIR}-${COMPONENT_NAME}/benchmark COMPONENT ${COMPONENT_NAME})
else()
    if (WIN32)
        install(TARGETS benchmark
            RUNTIME DESTINATION ${MAIN_DIR}-${WIN_RUN_X86_NAME}/benchmark COMPONENT ${WIN_RUN_X86_NAME})
    else ()
        install(TARGETS benchmark
            RUNTIME DESTINATION ${MAIN_DIR}-${RUN_X86_COMPONENT_NAME}/benchmark COMPONENT ${RUN_X86_COMPONENT_NAME})
    endif ()
endif()
