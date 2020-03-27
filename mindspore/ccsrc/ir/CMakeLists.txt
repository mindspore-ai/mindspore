file(GLOB_RECURSE _IR_ALL_SRC_FILES
      ./*.cc
      dtype/*.cc)

add_library(_mindspore_ir_obj OBJECT ${_IR_ALL_SRC_FILES})