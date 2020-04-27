file(GLOB_RECURSE _VM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(_mindspore_vm_obj OBJECT ${_VM_SRC_LIST})
