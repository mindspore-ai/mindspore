#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

# shellcheck disable=SC2034

set -e

init_default_options()
{
  # Init default values of build options
  export THREAD_NUM=8
  export DEBUG_MODE="off"
  VERBOSE=""
  export ENABLE_SECURITY="off"
  export ENABLE_COVERAGE="off"
  export RUN_TESTCASES="off"
  export RUN_CPP_ST_TESTS="off"
  export ENABLE_BACKEND=""
  export ENABLE_ASAN="off"
  export ENABLE_PROFILE="off"
  export INC_BUILD="off"
  export ENABLE_TIMELINE="off"
  export ENABLE_DUMP2PROTO="on"
  export ENABLE_DUMP_IR="on"
  export COMPILE_MINDDATA="on"
  export COMPILE_MINDDATA_LITE="lite_cv"
  export ENABLE_MPI="off"
  export CUDA_VERSION="10.1"
  export ASCEND_VERSION="910"
  export COMPILE_LITE="off"
  export LITE_PLATFORM=""
  export LITE_ENABLE_AAR="off"
  export USE_GLOG="on"
  export ENABLE_AKG="on"
  export ENABLE_ACL="off"
  export ENABLE_D="off"
  export ENABLE_DEBUGGER="on"
  export ENABLE_RDMA="off"
  export ENABLE_PYTHON="on"
  export ENABLE_GPU="off"
  export ENABLE_VERBOSE="off"
  export ENABLE_GITEE="off"
  export ENABLE_MAKE_CLEAN="off"
  export X86_64_SIMD="off"
  export ARM_SIMD="off"
  export DEVICE_VERSION=""
  export DEVICE=""
  export ENABLE_HIDDEN="on"
  export TENSORRT_HOME=""
  export USER_ENABLE_DUMP_IR=false
  export USER_ENABLE_DEBUGGER=false
  export ENABLE_SYM_FILE="off"
  export ENABLE_FAST_HASH_TABLE="on"
  export CUDA_ARCH="auto"
}
