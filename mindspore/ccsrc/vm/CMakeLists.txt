file(GLOB_RECURSE _VM_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_vm_obj OBJECT ${_VM_ALL_SRC_FILES})