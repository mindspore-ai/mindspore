file(GLOB_RECURSE _PRE_ACTIVE_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend/*.cc"
        "common/*.cc"
        "pass/*.cc"
        "gpu/*.cc"
        )

add_library(_mindspore_pre_active_obj OBJECT ${_PRE_ACTIVE_ALL_SRC_FILES})