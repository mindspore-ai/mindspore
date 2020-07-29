include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(converter_common_mid OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/node_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/storage.cc
        )
set_target_properties(converter_common_mid PROPERTIES COMPILE_FLAGS "-Wno-unused-function")
