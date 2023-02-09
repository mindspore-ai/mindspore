# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
if("x${CMAKE_CXX_COMPILER}" STREQUAL "x")
    set(CMAKE_CXX_COMPILER g++)
endif()
if("x${CMAKE_C_COMPILER}" STREQUAL "x")
    set(CMAKE_C_COMPILER gcc)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/caffe_plugin")
    add_subdirectory(caffe_plugin)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tf_plugin")
    add_subdirectory(tf_plugin)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tf_scope_fusion_pass")
    add_subdirectory(tf_scope_fusion_pass)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx_plugin")
    add_subdirectory(onnx_plugin)
endif()
