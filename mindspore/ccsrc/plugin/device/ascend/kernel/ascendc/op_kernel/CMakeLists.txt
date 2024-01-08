include_directories(${TOP_DIR}/graphengine/910/inc/external)

set(OP_KERNEL_DIR ${CMAKE_SOURCE_DIR}/mindspore/lite/tools/kernel_builder/ascend/ascendc/op_kernel)

foreach(compute_unit ${ASCEND_COMPUTE_UNIT})

  # generate aic-${compute_unit}-ops-info.json
  add_ops_info_target(
    TARGET
    ops_info_gen_${compute_unit}
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
    OPS_INFO
    ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
    INSTALL_DIR
    ${PKG_PATH}/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit})

  # generate ascendc impl py once
  if(NOT TARGET ascendc_impl_gen)
    add_ops_impl_target(
      TARGET
      ascendc_impl_gen
      OPS_INFO
      ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      IMPL_DIR
      ${OP_KERNEL_DIR}
      OUT_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/tbe
      INSTALL_DIR
      ${PKG_PATH}/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl)
  endif()
endforeach()

# install kernel file
if(${ENABLE_SOURCE_PACKAGE})
  file(GLOB KERNEL_FILES ${OP_KERNEL_DIR}/*.cpp ${OP_KERNEL_DIR}/*.h ${OP_KERNEL_DIR}/*.py)
  install(
    FILES ${KERNEL_FILES}
    DESTINATION
      ${PKG_PATH}/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
  )
endif()
