cmake_minimum_required(VERSION 3.14)
project(QuickStartC)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_C_COMPILER_VERSION} must not be less than 7.3.0")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(mindspore_quick_start_c main.c)

target_link_libraries(
        mindspore_quick_start_c
        -Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive
        pthread
        mindspore_glog
)

if(WIN32)
    target_link_libraries(
            mindspore_quick_start_c
            ssp
    )
else()
    target_link_libraries(
            mindspore_quick_start_c
            dl
    )
endif()
