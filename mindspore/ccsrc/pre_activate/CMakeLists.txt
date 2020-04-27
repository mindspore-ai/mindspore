file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "common/*.cc"
    "mem_reuse/*.cc"
    "pass/*.cc"
    "gpu/*.cc"
)

if (ENABLE_D)
    file(GLOB_RECURSE _D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ascend/*.cc")
	list(APPEND _PREACTIVATE_SRC_LIST ${_D_SRC_LIST})
endif ()

add_library(_mindspore_pre_activate_obj OBJECT ${_PREACTIVATE_SRC_LIST})