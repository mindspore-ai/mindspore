file(GLOB_RECURSE _PARALLEL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
if (ENABLE_DUMP_PROTO)
    list(REMOVE_ITEM _PARALLEL_SRC_FILES "parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
endif ()

set_property(SOURCE ${_PARALLEL_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PARALLEL)
add_library(_mindspore_parallel_obj OBJECT ${_PARALLEL_SRC_FILES})
