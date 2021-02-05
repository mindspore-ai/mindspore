file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

add_definitions(-DENABLE_DVPP_INTERFACE)

add_library(dvpp-utils OBJECT
        MDAclProcess.cc
        DvppCommon.cc
        ErrorCode.cpp
        ResourceManager.cc
        )
