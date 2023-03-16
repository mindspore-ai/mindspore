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

set -e

# check and set options
parse_device()
{
  if [[ "X$RUN_TESTCASES" == "Xon" && "X$DEVICE" != "X" ]]; then
    echo "WARNING:Option -e can't be set while option -t on/ut is set, reset device to empty."
    DEVICE=""
  fi

  # Parse device
  # Process build option
  export IFS_ORIGIN=$IFS
  export IFS=":"
  for D in $DEVICE;
  do
    if [[ "X$D" == "Xgpu" ]]; then
      export ENABLE_GPU="on"
      export GPU_BACKEND="cuda"
      ENABLE_CPU="on"
      ENABLE_MPI="on"
      # version default 10.1
      if [[ "X$DEVICE_VERSION" == "X" ]]; then
        DEVICE_VERSION=10.1
      fi
      if [[ "X$DEVICE_VERSION" != "X11.6" && "X$DEVICE_VERSION" != "X11.1" && "X$DEVICE_VERSION" != "X10.1" ]]; then
        echo "Invalid value ${DEVICE_VERSION} for option -V"
        usage
        exit 1
      fi
      export CUDA_VERSION="$DEVICE_VERSION"
      export ENABLE_AKG="on"
      export DEVICE_VERSION=
    elif [[ "X$D" == "Xrocm" ]]; then
      export ENABLE_GPU="on"
      export GPU_BACKEND="rocm"
      ENABLE_CPU="on"
      ENABLE_MPI="on"
      export ENABLE_AKG="off"
    elif [[ "X$D" == "Xd" || "X$D" == "Xascend" ]]; then
      # version default 910
      if [[ "X$DEVICE_VERSION" == "X" ]]; then
        DEVICE_VERSION=910
      fi
      # building 310 package by giving specific -V 310 instruction
      if [[ "X$DEVICE_VERSION" == "X310" ]]; then
        ENABLE_ACL="on"
        export ENABLE_AKG="off"
      # universal ascend package, building 910b package by giving specific -V 910b instruction
      elif [[ "X$DEVICE_VERSION" == "X910" || "X$DEVICE_VERSION" == "X910b" ]]; then
        export ENABLE_D="on"
        export ENABLE_ACL="on"
        ENABLE_CPU="on"
        export ENABLE_MPI="on"
        export ENABLE_AKG="on"
        export ASCEND_VERSION="$DEVICE_VERSION"
      else
        echo "Invalid value ${DEVICE_VERSION} for option -V"
        usage
        exit 1
      fi
      export DEVICE_VERSION=
    elif [[ "X$D" == "Xcpu" ]]; then
      export ENABLE_CPU="on"
      export ENABLE_AKG="on"
      export ENABLE_MPI="on"
    elif [[ "X$D" == "X" ]]; then
      :
    else
      echo "Invalid value ${DEVICE} for option -e"
      usage
      exit 1
    fi
  done
  export IFS=$IFS_ORIGIN
  if [[ "X$ENABLE_AKG" == "Xon" && "X$ENABLE_D" != "Xon" && "X$ENABLE_CPU" == "Xon" ]]; then
    # check llvm version for akg 
    HAS_LLVM=`bash ${BASEPATH}/scripts/build/akg_find_llvm.sh`
    export USE_LLVM=$HAS_LLVM
  fi
}
