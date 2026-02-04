# default build aclnn
add_subdirectory(aclnn)

# nnal is an optional package
if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    add_subdirectory(atb)
endif()

if(EXISTS ${ASCEND_NNAL_ASDSIP_PATH})
    add_subdirectory(asdsip)
endif()
