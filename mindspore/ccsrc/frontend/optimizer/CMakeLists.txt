file(GLOB_RECURSE _OPTIMIZER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${_OPTIMIZER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_OPTIMIZER)
add_library(_mindspore_frontend_optimizer_obj OBJECT ${_OPTIMIZER_SRC_FILES})
