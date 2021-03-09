cmake_minimum_required(VERSION 3.14)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

# Add directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add directory to linker search path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB_RECURSE QUICK_START_CXX ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(mindspore_quick_start_cpp ${QUICK_START_CXX})

target_link_libraries(
        mindspore_quick_start_cpp
        -Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive
        pthread
)

# Due to the increased compilation options for stack protection,
# it is necessary to target link ssp library when Use the static library in Windows.
if(WIN32)
    target_link_libraries(
            mindspore_quick_start_cpp
            ssp
    )
endif()