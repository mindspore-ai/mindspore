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

add_library(mindrt_mid OBJECT ${MINDRT_SRC})


