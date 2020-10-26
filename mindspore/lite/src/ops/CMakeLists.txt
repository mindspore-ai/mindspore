include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB OPS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/populate/*.cc
    )

add_library(cpu_ops_mid OBJECT ${OPS_SRC})
