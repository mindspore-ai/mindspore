# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )
cmake_minimum_required(VERSION 3.5.1)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

project(opp)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/cmake/config.cmake)
    include(${CMAKE_CURRENT_BINARY_DIR}/cmake/config.cmake)
else()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
    endif()
endif()

execute_process(COMMAND rm -rf ./makepkg)

message(STATUS "OP_COMMON_INC=${OP_COMMON_INC}")
# add sub compile directory
# open this when aicpu env is ok
#if(EXISTS "${CMAKE_SOURCE_DIR}/cpukernel")
#    add_subdirectory(cpukernel)
#endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/framework")
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/framework/mindspore")
        add_subdirectory(framework)
    endif()
endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/op_proto")
    add_subdirectory(op_proto)
endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/op_tiling")
    add_subdirectory(op_tiling)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/tbe")
    add_subdirectory(tbe)
endif()

if(UNIX)
    if(NOT "x$ENV{ADD_TESTCASE}" STREQUAL "x")
      # add testcases subdirectory
      if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/testcases")
        add_subdirectory(testcases)
      endif()
    endif()
endif()

message(STATUS "operation system is ${CMAKE_HOST_SYSTEM_NAME}")

if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/op_proto")
    set(ALL_MODULES ${OP_PROTO_TARGET})
endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/op_tiling")
    set(ALL_MODULES ${ALL_MODULES} ${OP_TILING_TARGET})
endif()
#if(EXISTS "${CMAKE_SOURCE_DIR}/cpukernel/CMakeLists.txt")
#    set(ALL_MODULES ${ALL_MODULES} ${AICPU_KERNEL_TARGET})
#    set(ALL_MODULES ${ALL_MODULES} ${AICPU_CONFIG_JSON_TARGET})
#endif()
message(STATUS "AICPU_CONFIG_INFO_FLAG=${AICPU_CONFIG_INFO_FLAG}")

if(EXISTS "${CMAKE_SOURCE_DIR}/tbe/CMakeLists.txt")
    set(ALL_MODULES ${ALL_MODULES} config_json)
endif()

if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/framework/caffe_plugin")
    set(ALL_MODULES ${ALL_MODULES} ${CAFFE_PLUGIN_TARGET})
endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/framework/tf_plugin")
    set(ALL_MODULES ${ALL_MODULES} ${TF_PLUGIN_TARGET})
endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/framework/tf_scope_fusion_pass")
    set(ALL_MODULES ${ALL_MODULES} ${TF_SCOPE_FUSION_PASS_TARGET})
endif()
if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/framework/onnx_plugin")
    set(ALL_MODULES ${ALL_MODULES} ${ONNX_PLUGIN_TARGET})
endif()

message(STATUS "ALL_MODULES=${ALL_MODULES}")
add_custom_target(${RUN_TARGET} ALL DEPENDS ${ALL_MODULES})

add_custom_command(TARGET ${RUN_TARGET}
        PRE_BUILD
        COMMAND mkdir -p ./makepkg/packages/${PROJECT_DIR}/fusion_rules/
        COMMAND mkdir -p ./makepkg/packages/${PROJECT_DIR}/op_impl/ai_core/tbe/mslite_impl
        COMMAND mkdir -p ./makepkg/packages/${PROJECT_DIR}/op_impl/vector_core/tbe/mslite_impl
        COMMAND mkdir -p ./makepkg/packages/${PROJECT_DIR}/op_impl/cpu/aicpu_kernel/impl
        COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/util/gen_impl_and_mrege_json.sh ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
        COMMAND cp ${CMAKE_SOURCE_DIR}/scripts/* ./makepkg/
        )

if(EXISTS ${CMAKE_SOURCE_DIR}/custom.proto)
    add_custom_command(TARGET ${RUN_TARGET}
        PRE_BUILD
        COMMAND cp ${CMAKE_SOURCE_DIR}/custom.proto ./makepkg/packages
        )
endif()

add_custom_command(TARGET ${RUN_TARGET}
        PRE_BUILD
        COMMAND chmod u+x ./makepkg/*.sh
        )

