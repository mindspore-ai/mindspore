file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)


if(ENABLE_PYTHON)
  add_library(callback OBJECT
          callback_manager.cc
          py_ds_callback.cc
          )
else()
  add_library(callback OBJECT
          callback_manager.cc
          )
endif()