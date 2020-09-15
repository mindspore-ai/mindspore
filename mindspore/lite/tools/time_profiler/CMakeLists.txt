# add shared link library

set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )

add_executable(timeprofiler
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/time_profiler.cc
        ${COMMON_SRC})

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(timeprofiler mindspore-lite)
else()
    target_link_libraries(timeprofiler mindspore-lite pthread)
endif()

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    install(TARGETS timeprofiler
            RUNTIME DESTINATION ${MAIN_DIR}-${COMPONENT_NAME}/time_profiler COMPONENT ${COMPONENT_NAME})
else()
    install(TARGETS timeprofiler
            RUNTIME DESTINATION ${MAIN_DIR}-${RUN_X86_COMPONENT_NAME}/time_profiler COMPONENT ${RUN_X86_COMPONENT_NAME})
endif()
