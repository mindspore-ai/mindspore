#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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
PROJECT_PATH=${BASEPATH}/../../..

# print usage message
usage()
{
  echo "Usage:"
  echo "sh runtests.sh [-e ascend310|ascend910] [-n testcase_name] [-d n]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -e Device target, default is ascend310"
  echo "    -d Device ID, default is 0"
  echo "    -n Run single tesecase, default off"
  echo "to be continued ..."
}

checkopts()
{
  DEVICE_TARGET_OPT="ascend310"
  DEVICE_ID_OPT=0
  TASECASE_NAME_OPT=""

  # Process the options
  while getopts 'he:d:n:' opt
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

cd ${PROJECT_PATH}/tests/st/cpp

MINDSPORE_PKG_PATH=`python -m pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
if [[ "X${MINDSPORE_PKG_PATH}" == "X" ]]; then
  MINDSPORE_PKG_PATH=${PROJECT_PATH}/build/package/mindspore:
fi

export LD_LIBRARY_PATH=${MINDSPORE_PKG_PATH}:${MINDSPORE_PKG_PATH}/lib:${PROJECT_PATH}/tests/st/cpp:$LD_LIBRARY_PATH
export GLOG_v=2
export GC_COLLECT_IN_CELL=1
export DEVICE_ID=$DEVICE_ID_OPT
if [[ "X$DEVICE_TARGET_OPT" == "Xascend310" ]]; then
  export DEVICE_TARGET=Ascend310
elif [[ "X$DEVICE_TARGET_OPT" == "Xascend910" ]]; then
  export DEVICE_TARGET=Ascend910
else
  export DEVICE_TARGET=$DEVICE_TARGET_OPT
fi

if [[ "X$TASECASE_NAME_OPT" != "X" ]]; then
  ./st_tests --gtest_filter=$TASECASE_NAME_OPT
else
  ./st_tests
fi
RET=$?
cd -

exit ${RET}
