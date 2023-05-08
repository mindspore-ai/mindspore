include_directories(${TOP_DIR})
include_directories(${TOP_DIR}/mindspore/lite)

file(GLOB RUNTIME_CONVERT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")
add_library(runtime_convert_plugin_mid OBJECT ${RUNTIME_CONVERT_SRC})

add_library(runtime_convert_plugin SHARED $<TARGET_OBJECTS:runtime_convert_plugin_mid>)
add_dependencies(runtime_convert_plugin fbs_inner_src)
target_link_libraries(runtime_convert_plugin mindspore-extendrt)

# for python
add_library(runtime_convert_plugin_python SHARED $<TARGET_OBJECTS:runtime_convert_plugin_mid>)
add_dependencies(runtime_convert_plugin_python fbs_inner_src)
target_link_libraries(runtime_convert_plugin_python mindspore-extendrt_python)


if(MSLITE_ENABLE_CONVERTER AND NOT WIN32)
    target_link_libraries(runtime_convert_plugin mindspore_converter)
    target_link_libraries(runtime_convert_plugin_python mindspore_converter)
endif()
