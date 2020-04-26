file(GLOB_RECURSE _PARALLEL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
if (ENABLE_DUMP_PROTO)
    list(REMOVE_ITEM _PARALLEL_SRC_FILES "parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
endif ()

add_library(_mindspore_parallel_obj OBJECT ${_PARALLEL_SRC_FILES})
