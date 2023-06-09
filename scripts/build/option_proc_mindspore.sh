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

build_option_proc_b()
{
  if [ "X$OPTARG" != "Xcpu" ]; then
    echo "Invalid value ${OPTARG} for option -b"
    usage
    exit 1
  fi
  ENABLE_BACKEND=$(echo "$OPTARG" | tr '[a-z]' '[A-Z]')
  if [[ "X$ENABLE_BACKEND" != "XCPU" ]]; then
    export ENABLE_CPU="on"
  fi
}

build_option_proc_l()
{
  check_on_off $OPTARG l
  export ENABLE_PYTHON="$OPTARG"
}

build_option_proc_s()
{
  check_on_off $OPTARG s
  if [[ "X$OPTARG" == "Xon" ]]; then
    if [[ "$USER_ENABLE_DUMP_IR" == true ]]; then
      echo "enable security, the dump ir is not available"
      usage
      exit 1
    fi
    if [[ "$USER_ENABLE_DEBUGGER" == true ]]; then
      echo "enable security, the debugger is not available"
      usage
      exit 1
    fi
    export ENABLE_DUMP_IR="off"
    export ENABLE_DEBUGGER="off"
  fi
  export ENABLE_SECURITY="$OPTARG"
  echo "enable security"
}

build_option_proc_upper_s()
{
  check_on_off $OPTARG S
  export ENABLE_GITEE="$OPTARG"
  echo "enable download from gitee"  
}

build_option_proc_upper_f()
{
  check_on_off $OPTARG F
  export ENABLE_FAST_HASH_TABLE="$OPTARG"
}

build_option_proc_z()
{
  eval ARG=\$\{$OPTIND\}
  if [[ -n "$ARG" && "$ARG" != -* ]]; then
    OPTARG="$ARG"
    check_on_off $OPTARG z
    OPTIND=$((OPTIND + 1))
  else
    OPTARG=""
  fi
  if [[ "X$OPTARG" == "Xoff" ]]; then
    export COMPILE_MINDDATA="off"
  fi
}

build_option_proc_upper_g()
{
  if [[ "X$OPTARG" == "Xcommon" || "X$OPTARG" == "Xauto" || "X$OPTARG" == "Xptx" ]]; then
    export CUDA_ARCH=$OPTARG
  else
    echo "Invalid value $OPTARG for option -G"
    usage
    exit 1
  fi
  echo "build gpu for arch $OPTARG"
}
