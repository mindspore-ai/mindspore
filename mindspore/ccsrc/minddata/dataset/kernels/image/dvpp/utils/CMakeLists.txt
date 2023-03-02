file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_definitions(-DENABLE_DVPP_INTERFACE)

set(DVPP_UTILS_SRC
    MDAclProcess.cc
    DvppCommon.cc
    ErrorCode.cpp
    ResourceManager.cc
    AclLiteUtils.cc
    VdecHelper.cc
    dvpp_video.cc
    acl_plugin.cc
    )
if(NOT MSLITE_ENABLE_ACL)
    set(DVPP_UTILS_SRC
    ${DVPP_UTILS_SRC}
    ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/graph/acl/acl_env_guard.cc
    )
endif()

add_library(dvpp_utils SHARED ${DVPP_UTILS_SRC})
enable_target_when_only_build_plugins(dvpp_utils)

if(MSLITE_ENABLE_ACL)
    find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    # find acl_env_guard in ascend_kernel_plugin
    target_link_libraries(dvpp_utils PRIVATE ascend_kernel_plugin minddata-lite ${acl} ${acl_dvpp} mindspore_core)
else()
    target_link_libraries(dvpp_utils PRIVATE _c_dataengine ${ACL} ${ACL_DVPP} mindspore_core mindspore_shared_lib)
endif()

if(MSLITE_ENABLE_CLOUD_MIND_DATA)
    add_dependencies(dvpp_utils fbs_src)
endif()
