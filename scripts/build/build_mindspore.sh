#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

# Create building path
build_mindspore()
{
    echo "start build mindspore project."
    mkdir -pv "${BUILD_PATH}/mindspore"
    cd "${BUILD_PATH}/mindspore"
    CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH"
    if [[ "X$ENABLE_COVERAGE" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_COVERAGE=ON"
    fi
    if [[ "X$RUN_TESTCASES" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TESTCASES=ON"
    fi
    if [[ "X$RUN_CPP_ST_TESTS" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CPP_ST=ON"
    fi
    if [[ -n "$ENABLE_BACKEND" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_${ENABLE_BACKEND}=ON"
    fi
    if [[ "X$ENABLE_SYM_FILE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_SYM_FILE=ON"
    fi
    if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
    fi
    if [[ "X$ENABLE_PROFILE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PROFILE=ON"
    fi
    if [[ "X$ENABLE_SECURITY" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_SECURITY=ON"
    fi
    if [[ "X$ENABLE_TIMELINE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TIMELINE=ON"
    fi
    if [[ "X$ENABLE_DUMP2PROTO" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_PROTO=ON"
    fi
    if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
    fi
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_IR=${ENABLE_DUMP_IR}"
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PYTHON=${ENABLE_PYTHON}"
    if [[ "X$ENABLE_MPI" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MPI=ON"
    fi
    if [[ "X$ENABLE_D" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON"
    fi
    if [[ "X$ENABLE_GPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON"
    fi
    if [[ "X$GPU_BACKEND" = "Xrocm" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DGPU_BACKEND_ROCM=ON -DROCM_PATH=$ROCM_PATH"
    fi
    if [[ "X$GPU_BACKEND" = "Xcuda" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS}  -DGPU_BACKEND_CUDA=ON -DUSE_CUDA=ON -DCUDA_PATH=$CUDA_PATH -DMS_REQUIRE_CUDA_VERSION=${CUDA_VERSION}"
    fi
    if [[ "X$ENABLE_CPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CPU=ON -DX86_64_SIMD=${X86_64_SIMD} -DARM_SIMD=${ARM_SIMD}"
    fi
    if [[ "X$COMPILE_MINDDATA" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MINDDATA=ON"
    fi
    if [[ "X$USE_GLOG" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DUSE_GLOG=ON"
    fi
    if [[ "X$ENABLE_AKG" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_AKG=ON"
        if [[ "X$USE_LLVM" = "Xon" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DUSE_LLVM=ON"
        fi
    fi
    if [[ "X$ENABLE_ACL" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ACL=ON"
    fi
    if [[ "X$ENABLE_DEBUGGER" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DEBUGGER=ON"
    fi

    if [[ "X$ENABLE_RDMA" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_RDMA=ON"
    fi
    if [[ "X$ENABLE_HIDDEN" = "Xoff" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_HIDDEN=OFF"
    fi
    if [[ "X$ENABLE_TRT" == "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DTENSORRT_HOME=${TENSORRT_HOME}"
    fi
    if [[ "X$ENABLE_FAST_HASH_TABLE" == "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_FAST_HASH_TABLE=ON"
    else
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_FAST_HASH_TABLE=OFF"
    fi
    if [[ "X$FASTER_BUILD_FOR_PLUGINS" == "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DONLY_BUILD_DEVICE_PLUGINS=ON"
    fi
    echo "${CMAKE_ARGS}"
    if [[ "X$INC_BUILD" = "Xoff" ]]; then
      cmake ${CMAKE_ARGS} ${BASEPATH}
    fi
    if [[ -n "$VERBOSE" ]]; then
      CMAKE_VERBOSE="--verbose"
    fi
    cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
    echo "success building mindspore project!"
}
