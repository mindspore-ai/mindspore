include_directories(${TOP_DIR})
include_directories(${TOP_DIR}/mindspore/lite)

file(GLOB RUNTIME_CONVERT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")
add_library(runtime_convert_plugin SHARED ${RUNTIME_CONVERT_SRC})
add_dependencies(runtime_convert_plugin fbs_inner_src)
target_link_libraries(runtime_convert_plugin mindspore-extendrt)

if(MSLITE_ENABLE_CONVERTER AND NOT WIN32)
    target_link_libraries(runtime_convert_plugin mindspore_converter)
endif()
