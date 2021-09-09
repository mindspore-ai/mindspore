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

build_option_proc_n()
{
  if [[ "X$OPTARG" == "Xoff" || "X$OPTARG" == "Xlite" || "X$OPTARG" == "Xfull" || "X$OPTARG" == "Xlite_cv"  || "X$OPTARG" == "Xwrapper" ]]; then
    export COMPILE_MINDDATA_LITE="$OPTARG"
  else
    echo "Invalid value ${OPTARG} for option -n"
    usage
    exit 1
  fi  
}

build_option_proc_upper_i()
{
  COMPILE_LITE="on"
  if [[ "$OPTARG" == "arm64" ]]; then
    LITE_PLATFORM="arm64"
  elif [[ "$OPTARG" == "arm32" ]]; then
    LITE_PLATFORM="arm32"
  elif [[ "$OPTARG" == "x86_64" ]]; then
    export LITE_PLATFORM="x86_64"
  else
    echo "-I parameter must be arm64、arm32 or x86_64"
    exit 1
  fi
}

build_option_proc_upper_a()
{
  export COMPILE_LITE="on"
  if [[ "$OPTARG" == "on" ]]; then
    export LITE_ENABLE_AAR="on"
  fi  
}

build_option_proc_upper_w()
{
  if [[ "$OPTARG" != "sse" && "$OPTARG" != "off" && "$OPTARG" != "avx" && "$OPTARG" != "avx512" && "$OPTARG" != "neon" ]]; then
    echo "Invalid value ${OPTARG} for option -W, -W parameter must be sse|neon|avx|avx512|off"
    usage
    exit 1
  fi
  if [[ "$OPTARG" == "sse" || "$OPTARG" == "avx" || "$OPTARG" == "avx512" ]]; then
    export X86_64_SIMD="$OPTARG"
  fi
  if [[ "$OPTARG" == "neon" ]]; then
    export ARM_SIMD="$OPTARG"
  fi  
}