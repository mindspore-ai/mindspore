#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
BASEPATH=$(cd "$(dirname $0)"; pwd)
PROJECT_PATH=${BASEPATH}/../../../..

# print usage message
usage()
{
  echo "Usage:"
  echo "sh runtests.sh [-e ascend310|ascend910|cpu] [-n testcase_name] [-d n]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -e Device target, default is cpu"
  echo "    -d Device ID, default is 0"
  echo "    -n Run single tesecase, default off"
  echo "to be continued ..."
}

checkopts()
{
  DEVICE_TARGET_OPT="cpu"
  DEVICE_ID_OPT=0
  TASECASE_NAME_OPT=""
  TEST_PATH=${PROJECT_PATH}/tests/st/cpp/c_api

  # Process the options
  while getopts 'h:e:d:n:' opt
  do
    case "${opt}" in
      h)
        usage
        exit 0
        ;;
      e)
        DEVICE_TARGET_OPT=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
        ;;
      d)
        DEVICE_ID_OPT=$OPTARG
        ;;
      n)
        TASECASE_NAME_OPT=$OPTARG
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

cd ${TEST_PATH}

OP_INFO_CFG_PATH=${PROJECT_PATH}/config/op_info.config
MINDSPORE_LIB_PATH=${PROJECT_PATH}/build/package/mindspore/lib/
C_API_LIB_PATH=${PROJECT_PATH}/build/mindspore/mindspore/ccsrc/c_api/
PYTHON_PATH=`which python`
PYTHON_LIB_PATH=${PYTHON_PATH%/*}/../lib

export LD_LIBRARY_PATH=${MINDSPORE_LIB_PATH}:${C_API_LIB_PATH}:${PYTHON_LIB_PATH}:${PROJECT_PATH}/tests/st/cpp/c_api:$LD_LIBRARY_PATH
export MINDSPORE_OP_INFO_PATH=${OP_INFO_CFG_PATH}
export GLOG_v=2
export GC_COLLECT_IN_CELL=1
export DEVICE_ID=$DEVICE_ID_OPT
if [[ "X$DEVICE_TARGET_OPT" == "Xascend" || "X$DEVICE_TARGET_OPT" == "XAscend" ]]; then
  export DEVICE_TARGET=Ascend
elif [[ "X$DEVICE_TARGET_OPT" == "Xcpu" ]]; then
  export DEVICE_TARGET=CPU
else
  export DEVICE_TARGET=$DEVICE_TARGET_OPT
fi

if [[ "X$TASECASE_NAME_OPT" != "X" ]]; then
  ./c_st_tests --gtest_filter=$TASECASE_NAME_OPT
else
  ./c_st_tests
fi
RET=$?
cd -

exit ${RET}
