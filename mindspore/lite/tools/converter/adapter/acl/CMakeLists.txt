include_directories(${TOP_DIR}/graphengine/metadef/inc/external)
include_directories(${TOP_DIR}/graphengine/inc)
include_directories(${TOP_DIR}/graphengine/inc/external)
include_directories(${TOP_DIR}/graphengine/ge)
include_directories(${TOP_DIR}/graphengine/metadef/inc)
include_directories(${TOP_DIR}/graphengine/inc/framework)
include_directories(${TOP_DIR}/graphengine/third_party/fwkacllib/inc)
include_directories(${TOP_DIR}/graphengine/third_party/fwkacllib/inc/toolchain)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB ACL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapper/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        )

set(ENABLE_ACL on)

add_subdirectory(${TOP_DIR}/mindspore/ccsrc/transform/graph_ir _mindspore_transform_graph_ir_obj)
add_subdirectory(${TOP_DIR}/mindspore/ccsrc/cxx_api mindspore_shared_lib)

set_property(SOURCE ${ACL_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(lite_acl_mid OBJECT ${ACL_SRC})

add_dependencies(lite_acl_mid fbs_inner_src)
