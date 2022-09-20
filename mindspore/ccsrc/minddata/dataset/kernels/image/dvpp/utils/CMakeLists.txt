file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_definitions(-DENABLE_DVPP_INTERFACE)

add_library(dvpp_utils SHARED
        MDAclProcess.cc
        DvppCommon.cc
        ErrorCode.cpp
        ResourceManager.cc
        AclLiteUtils.cc
        VdecHelper.cc
        dvpp_video.cc
        acl_plugin.cc
        )

target_link_libraries(dvpp_utils PRIVATE _c_dataengine ${ACL} ${ACL_DVPP} mindspore_core mindspore_shared_lib)

if(MSLITE_ENABLE_CLOUD_MIND_DATA)
    add_dependencies(dvpp_utils fbs_src)
endif()
