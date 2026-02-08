file(GLOB_RECURSE _ACL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(_mindspore_ops_kernel_ascend_aclop_obj OBJECT ${_ACL_SRC_LIST})