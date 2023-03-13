include_directories(${TOP_DIR}/graphengine/910/metadef/inc/external)
include_directories(${TOP_DIR}/graphengine/910/inc)
include_directories(${TOP_DIR}/graphengine/910/inc/external)
include_directories(${TOP_DIR}/graphengine/910/ge)
include_directories(${TOP_DIR}/graphengine/910/base)
include_directories(${TOP_DIR}/graphengine/910/metadef/inc)
include_directories(${TOP_DIR}/graphengine/910/inc/framework)
include_directories(${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc)
include_directories(${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc/aicpu)
include_directories(${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc/toolchain)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")

link_directories(${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
file(GLOB ACL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/api/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapper/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        ${TOP_DIR}/mindspore/lite/src/extendrt/utils/serialization.cc
        ${TOP_DIR}/mindspore/lite/src/extendrt/cxx_api/serialization.cc
        )

set(ACL_SRC ${ACL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/acl_pass.cc)

set(ENABLE_ACL on)
set(MODE_ASCEND_ACL off)
add_subdirectory(${TOP_DIR}/mindspore/ccsrc/transform/graph_ir _mindspore_transform_graph_ir_obj)
add_subdirectory(${TOP_DIR}/mindspore/ccsrc/cxx_api mindspore_shared_lib)


set_property(SOURCE ${ACL_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(ascend_pass_plugin SHARED ${ACL_SRC})
target_link_libraries(ascend_pass_plugin mindspore_shared_lib)
add_dependencies(ascend_pass_plugin mindspore_shared_lib)
add_dependencies(ascend_pass_plugin fbs_inner_src)
add_dependencies(ascend_pass_plugin mindspore_converter)
target_link_libraries(ascend_pass_plugin mindspore_converter)