if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH /usr/local/Ascend)
endif()
# driver
set(ASCEND_DRIVER_PATH ${ASCEND_PATH}/driver/lib64/common)
set(ASCEND_DRIVER_HAL_PATH ${ASCEND_PATH}/driver/lib64/driver)

# CANN packages
set(ASCEND_CANN_RUNTIME_PATH ${ASCEND_PATH}/latest/lib64)
set(ASCEND_CANN_OPP_PATH ${ASCEND_PATH}/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux)
set(ASCEND_CANN_OPP_AARCH64_PATH ${ASCEND_CANN_OPP_PATH}/aarch64)
set(ASCEND_CANN_OPP_X86_64_PATH ${ASCEND_CANN_OPP_PATH}/x86_64)
set(ASCEND_CANN_PLUGIN_PATH ${ASCEND_CANN_RUNTIME_PATH}/plugin/opskernel)

# Ascend-toolkit packages
set(ASCEND_TOOLKIT_RUNTIME_PATH ${ASCEND_PATH}/ascend-toolkit/latest/lib64)
set(ASCEND_TOOLKIT_OPP_PATH ${ASCEND_PATH}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux)
set(ASCEND_TOOLKIT_OPP_AARCH64_PATH ${ASCEND_TOOLKIT_OPP_PATH}/aarch64)
set(ASCEND_TOOLKIT_OPP_X86_64_PATH ${ASCEND_TOOLKIT_OPP_PATH}/x86_64)
set(ASCEND_TOOLKIT_PLUGIN_PATH ${ASCEND_TOOLKIT_RUNTIME_PATH}/plugin/opskernel)

# nnae packages (for rpath only)
set(ASCEND_NNAE_RUNTIME_PATH ${ASCEND_PATH}/nnae/latest/lib64)
set(ASCEND_NNAE_OPP_PATH ${ASCEND_PATH}/nnae/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling)
