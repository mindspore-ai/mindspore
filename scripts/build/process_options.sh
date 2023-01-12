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
process_options()
{
  # Process the options
  while getopts 'drvj:c:t:hb:s:a:g:p:ie:l:I:RP:D:zM:V:K:B:E:n:A:S:k:W:F:H:L:yG:f' opt
  do
    CASE_SENSIVE_ARG=${OPTARG}
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      d)
        DEBUG_MODE="on" ;;
      n)
        build_option_proc_n ;;
      y)
        export ENABLE_SYM_FILE="on" ;;
      r)
        export DEBUG_MODE="off" ;;
      v)
        build_option_proc_v ;;
      j)
        export THREAD_NUM=$OPTARG ;;
      c)
        build_option_proc_c ;;
      t)
        build_option_proc_t ;;
      g)
        build_option_proc_g ;;
      h)
        build_option_proc_h ;;
      b)
        build_option_proc_b ;;
      a)
        build_option_proc_a ;;
      p)
        build_option_proc_p ;;
      l)
        build_option_proc_l ;;
      i)
        export INC_BUILD="on" ;;
      s)
        build_option_proc_s ;;
      R)
        export ENABLE_TIMELINE="on"
        echo "enable time_line record" ;;
      S)
        build_option_proc_upper_s ;;
      k)
        check_on_off $OPTARG k
        export ENABLE_MAKE_CLEAN="$OPTARG"
        echo "enable make clean" ;;
      e)
        export DEVICE=$DEVICE:$OPTARG ;;
      M)
        check_on_off $OPTARG M
        export ENABLE_MPI="$OPTARG" ;;
      V)
        export DEVICE_VERSION=$OPTARG ;;
      P)
        check_on_off $OPTARG p
        export ENABLE_DUMP2PROTO="$OPTARG"
        echo "enable dump anf graph to proto file" ;;
      D)
        build_option_proc_upper_d ;;
      z)
        build_option_proc_z ;;
      I)
        build_option_proc_upper_i ;;
      K)
        check_on_off $OPTARG K
        export ENABLE_AKG="$OPTARG" ;;
      B)
        build_option_proc_upper_b ;;
      E)
        check_on_off $OPTARG E
        export ENABLE_RDMA="$OPTARG"
        echo "RDMA for RPC $ENABLE_RDMA" ;;
      A)
        build_option_proc_upper_a ;;
      W)
        build_option_proc_upper_w ;;
      F)
        build_option_proc_upper_f ;;
      H)
        check_on_off $OPTARG H
        export ENABLE_HIDDEN="$OPTARG"
        echo "${OPTARG} hidden" ;;
      L)
        export ENABLE_TRT="on"
        export TENSORRT_HOME="$CASE_SENSIVE_ARG"
        echo "Link Tensor-RT library. Path: ${CASE_SENSIVE_ARG}" ;;
      G)
        build_option_proc_upper_g ;;
      f)
        export FASTER_BUILD_FOR_PLUGINS="on" ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}
