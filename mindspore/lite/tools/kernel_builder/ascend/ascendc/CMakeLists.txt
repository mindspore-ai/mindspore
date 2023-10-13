set(CMAKE_BUILD_TYPE "Release")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2)
cmake_minimum_required(VERSION 3.5.1)
project(mslite_ascendc_opp)
include(${PROJECT_SOURCE_DIR}/cmake/config.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/func.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/intf.cmake)

execute_process(COMMAND mkdir -p ${PKG_PATH})
execute_process(COMMAND mkdir -p ${PKG_PATH}/packages)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
  add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
  add_subdirectory(op_kernel)
endif()

# modify vendor_name in install.sh and upgrade.sh
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/ascendc_scripts/install.sh
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/ascendc_scripts
  COMMAND cp -r ${PROJECT_SOURCE_DIR}/../scripts/* ${CMAKE_BINARY_DIR}/ascendc_scripts/
  COMMAND sed -i "s/vendor_name=customize/vendor_name=mslite_ascendc/g"
          ${CMAKE_BINARY_DIR}/ascendc_scripts/*)
add_custom_target(
  modify_vendor ALL DEPENDS ${CMAKE_BINARY_DIR}/ascendc_scripts/install.sh)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/ascendc_scripts/
  DESTINATION ${PKG_PATH}
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ)

install(
  FILES ${PROJECT_SOURCE_DIR}/custom.proto
  DESTINATION ${PKG_PATH}/packages
  OPTIONAL)

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(
  gen_version_info ALL
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_version_info.sh
          ${ASCEND_CANN_PACKAGE_PATH} ${CMAKE_CURRENT_BINARY_DIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.info
        DESTINATION ${PKG_PATH}/packages/vendors/${vendor_name}/)
