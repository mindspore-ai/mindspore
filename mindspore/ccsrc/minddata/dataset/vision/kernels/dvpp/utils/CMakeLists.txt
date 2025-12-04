file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_definitions(-DENABLE_DVPP_INTERFACE)

set(DVPP_UTILS_SRC
    # Ascend310
    MDAclProcess.cc
    DvppCommon.cc
    ErrorCode.cpp
    ResourceManager.cc
    AclLiteUtils.cc
    VdecHelper.cc
    dvpp_video.cc
    # plugin
    acl_plugin.cc
    )

set(DVPP_UTILS_SRC
    ${DVPP_UTILS_SRC}
    acl_env_guard.cc # in lite, src code has acl_env_guard.cc impl
    )

if(ENABLE_D)
set(DVPP_UTILS_SRC
    ${DVPP_UTILS_SRC}
    # Ascend910B
    dvpp_image_utils.cc
    dvpp_video_utils.cc
    )
endif()

add_library(dvpp_utils SHARED ${DVPP_UTILS_SRC})
enable_target_when_only_build_plugins(dvpp_utils)

find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(nnopbase libnnopbase.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl_dvpp_op libacl_dvpp_op.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl_dvpp_mpi libacl_dvpp_mpi.so ${ASCEND_CANN_RUNTIME_PATH})
target_link_libraries(dvpp_utils PRIVATE _c_dataengine ${acl} ${acl_dvpp} mindspore_core ${nnopbase} ${acl_dvpp_op}
  ${acl_dvpp_mpi})
add_dependencies(dvpp_utils _mindspore_ascend_symbol_obj)
target_link_libraries(dvpp_utils PRIVATE $<TARGET_OBJECTS:_mindspore_ascend_symbol_obj>)
