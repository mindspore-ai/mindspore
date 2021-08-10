cmake_minimum_required(VERSION 3.5.2)
project(yolov4tinypost)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(PLUGIN_NAME "yolov4tiny_mindspore_post")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(ACL_LIB_PATH $ENV{ASCEND_HOME}/nnrt/latest/acllib)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories($ENV{MX_SDK_HOME}/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include/opencv4)
include_directories($ENV{MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories($ENV{MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories($ENV{MX_SDK_HOME}/opensource/lib/glib-2.0/include)

link_directories($ENV{MX_SDK_HOME}/lib)
link_directories($ENV{MX_SDK_HOME}/opensource/lib/)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_compile_options("-Dgoogle=mindxsdk_private")

add_definitions(-DENABLE_DVPP_INTERFACE)

#message("ACL_LIB_PATH:${ACL_LIB_PATH}.")
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ACL_LIB_PATH}/include)

add_library(${TARGET_LIBRARY} SHARED ../../mxbase/src/PostProcess/Yolov4TinyMindsporePost.cpp 
../../mxbase/src/PostProcess/Yolov4TinyMindsporePost.h)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxpidatatype mxbase)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION $ENV{MX_SDK_HOME}/lib/modelpostprocessors/)