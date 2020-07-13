file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(engine-opt OBJECT
          pass.cc
          post/repeat_pass.cc
          pre/cache_pass.cc
          pre/cache_transform_pass.cc
          pre/removal_nodes.cc
          pre/removal_pass.cc
          util/printer_pass.cc
        )
