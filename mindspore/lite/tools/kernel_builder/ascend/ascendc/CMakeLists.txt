project(opp)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
  add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
  add_subdirectory(op_kernel)
endif()

# modify vendor_name in install.sh and upgrade.sh
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/scripts/install.sh
          ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/scripts
  COMMAND cp -r ${CMAKE_SOURCE_DIR}/scripts/* ${CMAKE_BINARY_DIR}/scripts/
  COMMAND sed -i "s/vendor_name=customize/vendor_name=${vendor_name}/g"
          ${CMAKE_BINARY_DIR}/scripts/*)
add_custom_target(
  modify_vendor ALL DEPENDS ${CMAKE_BINARY_DIR}/scripts/install.sh
                            ${CMAKE_BINARY_DIR}/scripts/upgrade.sh)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/scripts/
  DESTINATION .
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ)

install(
  FILES ${CMAKE_SOURCE_DIR}/custom.proto
  DESTINATION packages
  OPTIONAL)

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(
  gen_version_info ALL
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_version_info.sh
          ${ASCEND_CANN_PACKAGE_PATH} ${CMAKE_CURRENT_BINARY_DIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.info
        DESTINATION packages/vendors/${vendor_name}/)

# CPack config
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION "CPack opp project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack opp project")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME "custom_opp_${SYSTEM_INFO}.run")
set(CPACK_GENERATOR External)
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
set(CPACK_EXTERNAL_BUILT_PACKAGES
    ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME}
)
include(CPack)
