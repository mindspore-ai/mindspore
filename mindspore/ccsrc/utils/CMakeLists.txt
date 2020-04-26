file(GLOB_RECURSE _UTILS_SRC_LIST ./*.cc)

if (NOT ENABLE_GE)
    file(GLOB_RECURSE _UTILS_GE_SRC_FILES ./callbacks_ge.cc)
    list(REMOVE_ITEM _UTILS_SRC_LIST ${_UTILS_GE_SRC_FILES})
endif ()

add_library(_mindspore_utils_obj OBJECT ${_UTILS_SRC_LIST})
