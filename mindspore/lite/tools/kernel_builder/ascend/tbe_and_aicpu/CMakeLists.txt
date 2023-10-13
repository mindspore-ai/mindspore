# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2)
cmake_minimum_required(VERSION 3.5.1)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

set(ASCEND_HOME $ENV{ASCEND_HOME})

project(mslite_ascend_tbe_and_aicpu_opp)
set(VENDOR_NAME "mslite_tbe_and_aicpu")

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/cmake/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/cmake/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
  endif()
endif()

execute_process(COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/makepkg)

message(STATUS "OP_COMMON_INC=${OP_COMMON_INC}")
# add sub compile directory open this when aicpu env is ok if(EXISTS
# "${CMAKE_CURRENT_SOURCE_DIR}/cpukernel") add_subdirectory(cpukernel) endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/op_proto")
  add_subdirectory(op_proto)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tbe")
  add_subdirectory(tbe)
endif()

message(STATUS "operation system is ${CMAKE_HOST_SYSTEM_NAME}")

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/op_proto")
  set(ALL_MODULES ${OP_PROTO_TARGET})
endif()
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpukernel/CMakeLists.txt")
# set(ALL_MODULES ${ALL_MODULES} ${AICPU_KERNEL_TARGET}) set(ALL_MODULES
# ${ALL_MODULES} ${AICPU_CONFIG_JSON_TARGET}) endif()
message(STATUS "AICPU_CONFIG_INFO_FLAG=${AICPU_CONFIG_INFO_FLAG}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tbe/CMakeLists.txt")
  set(ALL_MODULES ${ALL_MODULES} config_json)
endif()

message(STATUS "ALL_MODULES=${ALL_MODULES}")
add_custom_target(${RUN_TARGET} ALL DEPENDS ${ALL_MODULES})
message("RUN_TARGET ${RUN_TARGET}")

set(VENDOR_IMPL "${VENDOR_NAME}_impl")
add_custom_command(
  TARGET ${RUN_TARGET}
  PRE_BUILD
  COMMAND
    mkdir -p
    ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/${PROJECT_DIR}/op_impl/ai_core/tbe/${VENDOR_IMPL}
  COMMAND
    mkdir -p
    ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/${PROJECT_DIR}/op_impl/vector_core/tbe/${VENDOR_IMPL}
  COMMAND
    mkdir -p
    ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/${PROJECT_DIR}/op_impl/cpu/aicpu_kernel/impl
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_impl_and_mrege_json.sh
          ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/* ./makepkg/
  COMMAND sed -i "s/vendor_name=customize/vendor_name=mslite_tbe_and_aicpu/g" ./makepkg/install.sh
  COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/../akg/gen_custom_op_files.py
    ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/${PROJECT_DIR}/op_impl/ai_core/tbe/config
    ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/${PROJECT_DIR}/op_impl/ai_core/tbe/${VENDOR_IMPL}
)

add_custom_command(
  TARGET ${RUN_TARGET}
  PRE_BUILD
  COMMAND chmod u+x ${CMAKE_CURRENT_BINARY_DIR}/makepkg/*.sh)
