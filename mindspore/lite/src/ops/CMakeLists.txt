file(GLOB_RECURSE C_OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(c_ops_mid OBJECT ${C_OPS_SRC})