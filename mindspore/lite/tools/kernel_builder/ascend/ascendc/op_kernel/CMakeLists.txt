include_directories(${TOP_DIR}/graphengine/910/inc/external)
include_directories(${TOP_DIR}/graphengine/910b/inc/external)

set(tikreplaylib_DIR
    ${ASCEND_TENSOR_COMPILER_PATH}/tikcpp/tikreplaylib/lib/cmake)
find_package(tikreplaylib REQUIRED)
message(STATUS "PACKAGE tikreplaylib FOUND")

# replay config
set(BATCH_MODE_REPLAY_LIST)
set(ITERATOR_MODE_REPLAY_LIST)

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
    ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit})

  # generate ascendc impl py once
  if(NOT TARGET ascendc_impl_gen)
    add_ops_impl_target(
      TARGET
      ascendc_impl_gen
      OPS_INFO
      ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      OPS_BATCH
      ${BATCH_MODE_REPLAY_LIST}
      OPS_ITERATE
      ${ITERATOR_MODE_REPLAY_LIST}
      IMPL_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}
      OUT_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/tbe
      INSTALL_DIR
      ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl)
  endif()

  # dynamic shape binary compile
  if(${ENABLE_BINARY_PACKAGE})
    add_bin_compile_target(
      TARGET
      ascendc_bin_${compute_unit}
      OPS_INFO
      ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
      IMPL_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}
      ADP_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/tbe/dynamic
      OUT_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
      INSTALL_DIR
      ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel
      COMPUTE_UNIT
      ${compute_unit})
    add_dependencies(ascendc_bin_${compute_unit} ascendc_impl_gen)
  endif()

  # generate replay <op>_impl.cpp, <op>_replay.cpp and <op>_entry.cce
  add_ops_replay_targets(
    OPS_BATCH
    ${BATCH_MODE_REPLAY_LIST}
    OPS_ITERATE
    ${ITERATOR_MODE_REPLAY_LIST}
    OPS_INFO
    ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
    IMPL_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    OUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/tbe/replay/${compute_unit}
    INSTALL_DIR
    ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_replay
    COMPUTE_UNIT
    ${compute_unit})
endforeach()

# generate npu_supported_ops.json
add_npu_support_target(
  TARGET
  npu_supported_ops
  OPS_INFO_DIR
  ${ASCEND_AUTOGEN_PATH}
  OUT_DIR
  ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
  INSTALL_DIR
  ${PKG_PATH}/packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE})

if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
  add_subdirectory(testcases)
endif()

# install kernel file
if(${ENABLE_SOURCE_PACKAGE})
  file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
  install(
    FILES ${KERNEL_FILES}
    DESTINATION
      ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
  )
endif()
