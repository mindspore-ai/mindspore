if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(KERNEL_OPC_PATH $ENV{ASCEND_CUSTOM_PATH}/latest/)
elseif(EXISTS /usr/local/Ascend/latest)
    set(KERNEL_OPC_PATH /usr/local/Ascend/latest)
elseif(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
    set(KERNEL_OPC_PATH /usr/local/Ascend/ascend-toolkit/latest)
else()
    set(KERNEL_OPC_PATH /usr/local/Ascend)
endif()
set(OPC_PATH ${KERNEL_OPC_PATH}/aarch64-linux/bin/opc)
set(BUILTIN_OPP_PATH ${KERNEL_OPC_PATH}/opp/built-in/op_impl/ai_core/tbe)

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
      ${CMAKE_CURRENT_SOURCE_DIR}
      OUT_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/tbe
      INSTALL_DIR
      ${PKG_PATH}/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl)
  endif()

  # dynamic shape binary compile
  if(NOT EXISTS ${OPC_PATH})
    message("${OPC_PATH} isn't exist! Please check!")
  endif()

  if(${ENABLE_BINARY_PACKAGE} AND NOT ${ENABLE_CROSS_COMPILE} AND EXISTS ${OPC_PATH}
    AND ${compute_unit} STREQUAL "ascend910b")
    add_bin_compile_target(TARGET ascendc_bin_${compute_unit}
      OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      ADP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/dynamic
      OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
      INSTALL_DIR ${PKG_PATH}/${vendor_name}/op_impl/ai_core/tbe/kernel
      COMPUTE_UNIT ${compute_unit}
      OPC_PATH ${OPC_PATH}
      OPP_PATH ${BUILTIN_OPP_PATH})
    add_dependencies(ascendc_bin_${compute_unit} ascendc_impl_gen)
  endif()

  if(${ENABLE_CROSS_COMPILE} AND ${ENABLE_BINARY_PACKAGE})
    add_cross_compile_target(
      TARGET bin_${compute_unit}
      OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernel
      INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/)
  endif()
endforeach()

# install kernel file
if(${ENABLE_SOURCE_PACKAGE})
  file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
  install(
    FILES ${KERNEL_FILES}
    DESTINATION
      ${PKG_PATH}/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
  )
endif()
