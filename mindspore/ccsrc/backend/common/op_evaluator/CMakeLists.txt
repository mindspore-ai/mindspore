file(GLOB_RECURSE _OP_EVAL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )
list(APPEND _GRAPH_KERNEL_SRC_FILES ${_OP_EVAL_SRC_LIST})

add_library(_mindspore_backend_common_op_evaluator_obj OBJECT ${_GRAPH_KERNEL_SRC_FILES})
