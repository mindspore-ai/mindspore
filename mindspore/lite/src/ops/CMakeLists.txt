file(GLOB_RECURSE OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(ops_mid_ OBJECT ${OPS_SRC})