file(GLOB _ASCEND_GE_EXECUTOR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
# avoid cpp error as CANN is deprecating current version of ops protos by Dec 2026
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=cpp")
set_property(SOURCE ${_ASCEND_GE_EXECUTOR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_ge_executor_obj OBJECT ${_ASCEND_GE_EXECUTOR_SRC_LIST})
