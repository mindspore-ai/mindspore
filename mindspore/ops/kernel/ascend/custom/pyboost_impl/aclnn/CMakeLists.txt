file(GLOB_RECURSE EXTENSION_ASCEND_ACLNN_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)

set_property(SOURCE ${EXTENSION_ASCEND_ACLNN_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_EXTENSION)
add_library(mindspore_extension_ascend_aclnn STATIC ${EXTENSION_ASCEND_ACLNN_SRC_LIST})

target_link_libraries(mindspore_extension_ascend_aclnn PRIVATE mindspore_extension)
