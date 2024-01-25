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

build_option_proc_v()
{
  export ENABLE_VERBOSE="on"
  export VERBOSE="VERBOSE=1"
}

build_option_proc_c()
{
  check_on_off $OPTARG c
  export ENABLE_COVERAGE="$OPTARG"
}

build_option_proc_t()
{
  if [[ "X$OPTARG" == "Xon" || "X$OPTARG" == "Xut" ]]; then
    export RUN_TESTCASES="on"
  elif [[ "X$OPTARG" == "Xoff" ]]; then
    export RUN_TESTCASES="off"
  elif [[ "X$OPTARG" == "Xst" ]]; then
    export RUN_CPP_ST_TESTS="on"
  else
    echo "Invalid value ${OPTARG} for option -t"
    usage
    exit 1
  fi
}

build_option_proc_g()
{
  check_on_off $OPTARG g
  export USE_GLOG="$OPTARG" 
}

build_option_proc_h()
{
  usage
  exit 0
}

build_option_proc_a()
{
  check_on_off $OPTARG a
  export ENABLE_ASAN="$OPTARG"
}

build_option_proc_p()
{
  check_on_off $OPTARG p
  export ENABLE_PROFILE="$OPTARG"
}

build_option_proc_upper_d()
{
  check_on_off $OPTARG D
  if [[ "X$OPTARG" == "Xon" ]]; then
    if [[ "X$ENABLE_SECURITY" == "Xon" ]]; then
      echo "enable security, the dump ir is not available"
      usage
      exit 1
    fi
    export USER_ENABLE_DUMP_IR=true
  fi
  export ENABLE_DUMP_IR="$OPTARG"
  echo "enable dump function graph ir"
}

build_option_proc_upper_b()
{
  check_on_off $OPTARG B
  if [[ "X$OPTARG" == "Xon" ]]; then
    if [[ "X$ENABLE_SECURITY" == "Xon" ]]; then
      echo "enable security, the debugger is not available"
      usage
      exit 1
    fi
    export USER_ENABLE_DEBUGGER=true
  fi
  export ENABLE_DEBUGGER="$OPTARG"
}