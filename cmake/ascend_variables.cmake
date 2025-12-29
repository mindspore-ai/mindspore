# default cann install path: /usr/local/Ascend
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH /usr/local/Ascend)
endif()
# driver
set(ASCEND_DRIVER_PATH ${ASCEND_PATH}/driver/lib64/common)
set(ASCEND_DRIVER_HAL_PATH ${ASCEND_PATH}/driver/lib64/driver)

# CANN binary search paths
set(ASCEND_CANN_RUNTIME_PATH ${ASCEND_PATH}/cann/lib64)
set(ASCEND_CANN_OPP_PATH ${ASCEND_PATH}/cann/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux)
set(ASCEND_CANN_OPP_AARCH64_PATH ${ASCEND_CANN_OPP_PATH}/aarch64)
set(ASCEND_CANN_OPP_X86_64_PATH ${ASCEND_CANN_OPP_PATH}/x86_64)
set(ASCEND_CANN_PLUGIN_PATH ${ASCEND_CANN_RUNTIME_PATH}/plugin/opskernel)
set(ASCEND_CANN_AICPU_KERNEL_PATH ${ASCEND_PATH}/cann/opp/built-in/op_impl/aicpu/aicpu_kernel)

# nnal packages (for ATB kernel and ASDSIP kernel)
set(ASCEND_NNAL_RUNTIME_PATH ${ASCEND_PATH}/nnal/)

# use cxx_abi=0
if(NOT ENABLE_GLIBCXX)
    set(ASCEND_NNAL_ATB_PATH ${ASCEND_NNAL_RUNTIME_PATH}/atb/latest/atb/cxx_abi_0/)
    set(ASCEND_NNAL_ATB_OPP_PATH ${ASCEND_NNAL_RUNTIME_PATH}/atb/latest/atb/cxx_abi_0/lib/)
    if(EXISTS ${ASCEND_NNAL_ATB_PATH})
        add_compile_definitions(ENABLE_ATB)
    endif()
endif()

set(ASCEND_NNAL_ASDSIP_PATH ${ASCEND_NNAL_RUNTIME_PATH}/asdsip/latest/)
