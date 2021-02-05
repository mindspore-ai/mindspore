file(GLOB_RECURSE _UTILS_SRC_LIST ./*.cc)

if(NOT ENABLE_GE)
    file(GLOB_RECURSE _UTILS_GE_SRC_FILES ./callbacks_ge.cc)
    list(REMOVE_ITEM _UTILS_SRC_LIST ${_UTILS_GE_SRC_FILES})
endif()
if(NOT ENABLE_D AND NOT ENABLE_TESTCASES)
    file(GLOB_RECURSE _UTILS_D_SRC_FILES ./runtime_error_codes.cc)
    list(REMOVE_ITEM _UTILS_SRC_LIST ${_UTILS_D_SRC_FILES})
endif()

set_property(SOURCE ${_UTILS_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_UTILS)
add_library(_mindspore_utils_obj OBJECT ${_UTILS_SRC_LIST})
