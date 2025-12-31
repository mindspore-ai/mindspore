# default cann install path: /usr/local/Ascend
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH_BASE $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH_BASE /usr/local/Ascend)
endif()

# for CANN 8.5.0 and above, the default install path is /usr/local/Ascend/cann
if(EXISTS ${ASCEND_PATH_BASE}/cann)
    set(ASCEND_PATH ${ASCEND_PATH_BASE}/cann)
    message("Compiling MindSpore with CANN 8.5.0 and above.")
elseif(EXISTS ${ASCEND_PATH_BASE}/ascend-toolkit/latest)
    set(ASCEND_PATH ${ASCEND_PATH_BASE}/ascend-toolkit/latest)
    message("Compiling MindSpore with CANN 8.3.0 and below.")
else()
    message(FATAL_ERROR "Can not find CANN installation path in the environment."
                        "Please check if the cann package is installed properly.")
endif()
message("ASCEND_PATH is set to: ${ASCEND_PATH}")

# CANN binary search paths
set(ASCEND_CANN_RUNTIME_PATH ${ASCEND_PATH}/lib64)
set(ASCEND_CANN_OPP_PATH ${ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux)
set(ASCEND_CANN_OPP_AARCH64_PATH ${ASCEND_CANN_OPP_PATH}/aarch64)
set(ASCEND_CANN_OPP_X86_64_PATH ${ASCEND_CANN_OPP_PATH}/x86_64)
set(ASCEND_CANN_PLUGIN_PATH ${ASCEND_CANN_RUNTIME_PATH}/plugin/opskernel)
set(ASCEND_CANN_AICPU_KERNEL_PATH ${ASCEND_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)

# nnal packages (for ATB kernel and ASDSIP kernel)
set(ASCEND_NNAL_RUNTIME_PATH ${ASCEND_PATH_BASE}/nnal/)

# use cxx_abi=0
if(NOT ENABLE_GLIBCXX)
    set(ASCEND_NNAL_ATB_PATH ${ASCEND_NNAL_RUNTIME_PATH}/atb/latest/atb/cxx_abi_0/)
    set(ASCEND_NNAL_ATB_OPP_PATH ${ASCEND_NNAL_RUNTIME_PATH}/atb/latest/atb/cxx_abi_0/lib/)
    if(EXISTS ${ASCEND_NNAL_ATB_PATH})
        add_compile_definitions(ENABLE_ATB)
    endif()
endif()

set(ASCEND_NNAL_ASDSIP_PATH ${ASCEND_NNAL_RUNTIME_PATH}/asdsip/latest/)
