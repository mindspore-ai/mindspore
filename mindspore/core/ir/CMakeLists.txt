file(GLOB_RECURSE _IR_SRC_LIST ./*.cc dtype/*.cc)
set_property(SOURCE ${_IR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_IR)
add_library(_mindspore_ir_obj OBJECT ${_IR_SRC_LIST})
