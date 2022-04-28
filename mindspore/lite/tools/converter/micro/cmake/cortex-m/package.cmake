include(CMakePackageConfigHelpers)

set(RUNTIME_PKG_NAME ${PKG_NAME_PREFIX}-${RUNTIME_COMPONENT_NAME})

include(${TOP_DIR}/cmake/package_micro.cmake)

__install_micro_wrapper()
__install_micro_codegen()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL ${RUNTIME_COMPONENT_NAME})
set(CPACK_PACKAGE_FILE_NAME ${PKG_NAME_PREFIX})

if(WIN32)
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output)
else()
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output/tmp)
endif()
set(CPACK_PACKAGE_CHECKSUM SHA256)
include(CPack)
