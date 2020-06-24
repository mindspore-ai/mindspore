file(GLOB_RECURSE _IR_SRC_LIST ./*.cc dtype/*.cc)
file(GLOB_RECURSE _IR_LITE_SRC_FILES
        ./lite/tensor.cc
        )
list(REMOVE_ITEM _IR_SRC_LIST ${_IR_LITE_SRC_FILES})
set_property(SOURCE ${_IR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_IR)
add_library(_mindspore_ir_obj OBJECT ${_IR_SRC_LIST})
