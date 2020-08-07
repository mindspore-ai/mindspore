# add shared link library

set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )

add_executable(timeprofile
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/time_profile.cc
        ${COMMON_SRC})

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(timeprofile mindspore-lite)
else()
    target_link_libraries(timeprofile mindspore-lite pthread)
endif()
