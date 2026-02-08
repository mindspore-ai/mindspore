cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE OP_PLUGIN_MOCK_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mindspore_op_plugin_mock SHARED ${OP_PLUGIN_MOCK_SRC_LIST})

set_target_properties(mindspore_op_plugin_mock PROPERTIES
  CXX_VISIBILITY_PRESET default
)

if(WIN32)
  set_target_properties(mindspore_op_plugin_mock PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
