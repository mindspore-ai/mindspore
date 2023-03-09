include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

file(GLOB MINDRT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/actor/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/async/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/evloop/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/timer/*.cc
    )

set(MINDRT_SRC ${MINDRT_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thread/threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thread/actor_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thread/parallel_thread_pool_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thread/parallel_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thread/core_affinity.cc
    )

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_compile_definitions(BUILDING_CORE_DLL)
endif()

add_library(mindrt_mid OBJECT ${MINDRT_SRC})
