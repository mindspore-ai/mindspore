file(GLOB_RECURSE _IR_SRC_LIST ./*.cc dtype/*.cc)
add_library(_mindspore_ir_obj OBJECT ${_IR_SRC_LIST})
