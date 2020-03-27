file(GLOB_RECURSE _OPTIMIZER_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

add_library(_mindspore_optimizer_obj OBJECT ${_OPTIMIZER_ALL_SRC_FILES})
if(ENABLE_DUMP_PROTO)
    file(GLOB_RECURSE _PROTO_SRC_LIST  "parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
    target_sources(_mindspore_optimizer_obj  PRIVATE ${_PROTO_SRC_LIST})
endif()