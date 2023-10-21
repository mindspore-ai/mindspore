cmake_minimum_required(VERSION 3.12)
# recommended usage by the official documents.
project(triton-mslite-backend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_GITEE "Enable to fetch triton from gitee" OFF)
option(TRITON_RAPID_JSON_PATH "Enable to fetch triton from gitee" OFF)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ENV{MINDSPORE_LITE_PKG_ROOT_PATH})
  set(MINDSPORE_LITE_PKG_ROOT_PATH $ENV{MINDSPORE_LITE_PKG_ROOT_PATH})
else()
  message(FATAL_ERROR "Set the MindSpore Lite package path by the environment MINDSPORE_LITE_PKG_ROOT_PATH")
endif()
if(DEFINED ENV{TRITON_ENABLE_GITEE})
  set(TRITON_ENABLE_GITEE $ENV{TRITON_ENABLE_GITEE})
endif()

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_mslite.ldscript libtriton_mslite.ldscript COPYONLY)

file(GLOB SRC_FILES src/*.cc)
add_library(
  triton-mslite-backend SHARED
  ${SRC_FILES}
)

add_library(
  TritonMSLiteBackend::triton-mslite-backend ALIAS triton-mslite-backend
)


# Dependencies
if(NOT ${TRITON_ENABLE_GITEE})
  include(FetchContent)

  FetchContent_Declare(
    repo-common
    GIT_REPOSITORY https://github.com/triton-inference-server/common.git
    GIT_TAG ${TRITON_COMMON_REPO_TAG}
    GIT_SHALLOW ON
  )
  FetchContent_Declare(
    repo-core
    GIT_REPOSITORY https://github.com/triton-inference-server/core.git
    GIT_TAG ${TRITON_CORE_REPO_TAG}
    GIT_SHALLOW ON
  )
  FetchContent_Declare(
    repo-backend
    GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
    GIT_TAG ${TRITON_BACKEND_REPO_TAG}
    GIT_SHALLOW ON
  )
  FetchContent_MakeAvailable(repo-common repo-core repo-backend)
else()
  string(REPLACE "/mindspore/lite/tools/providers/triton/backend" "" TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  find_package(Patch)
  if(NOT Patch_FOUND)
    message(FATAL_ERROR "Patch not found, please set environment variable MS_PATCH_PATH to path "
            "where Patch is located, usually found in GIT_PATH/usr/bin on Windows")
  endif()

  if(TRITON_RAPID_JSON_PATH)
    set(TRITON_RAPID_JSON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TRITON_RAPID_JSON_PATH})
  endif()

  include(${TOP_DIR}/cmake/utils.cmake)
  include(${TOP_DIR}/cmake/external_libs/triton_backend.cmake)
  target_include_directories(
    triton-mslite-backend
    PRIVATE
    ${TRITON_RAPID_JSON_PATH}
  )
endif()

target_include_directories(
  triton-mslite-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MINDSPORE_LITE_PKG_ROOT_PATH}/runtime
)

target_compile_features(triton-mslite-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-mslite-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror -fstack-protector-all -fPIC -D_FORTIFY_SOURCE=2 -O2>
)

target_link_options(triton-mslite-backend PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack -fstack-protector-all -s)
if(NOT ${TRITON_ENABLE_GITEE})
  target_link_libraries(
    triton-mslite-backend
    PRIVATE
      triton-core-serverstub  # from repo-core
      triton-backend-utils    # from repo-backend
  )
else()
  target_link_libraries(
    triton-mslite-backend
    PRIVATE
      mindspore::tritonasyncworkqueue  # from repo-common
      mindspore::tritonserver          # from repo-core
      mindspore::tritonbackendutils    # from repo-backend
  )
endif()

target_link_directories(
  triton-mslite-backend
  PRIVATE
    ${MINDSPORE_LITE_PKG_ROOT_PATH}/runtime/lib
)

target_link_libraries(
  triton-mslite-backend
  PRIVATE
    mindspore-lite
    pthread
    dl
)

set_target_properties(
  triton-mslite-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_mslite
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_mslite.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_mslite.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonMSLiteBackend)

install(
  TARGETS
    triton-mslite-backend
  EXPORT
    triton-mslite-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/mslite
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/mslite
)

install(
  EXPORT
    triton-mslite-backend-targets
  FILE
    TritonMSLiteBackendTargets.cmake
  NAMESPACE
    TritonMSLiteBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonMSLiteBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonMSLiteBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonMSLiteBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-mslite-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonMSLiteBackendTargets.cmake
  NAMESPACE TritonMSLiteBackend::
)

export(PACKAGE TritonMSLiteBackend)
