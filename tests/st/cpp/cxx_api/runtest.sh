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
PROJECT_PATH=${BASEPATH}/../../../..

# print usage message
usage()
{
  echo "Usage:"
  echo "sh runtests.sh [-e ascend310|ascend910] [-n testcase_name] [-d n] [-t cpp|python] [-r path]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -e Device target, default is ascend310"
  echo "    -d Device ID, default is 0"
  echo "    -n Run single tesecase, default off"
  echo "    -t Type of MindSpore package to be tested, default is cpp"
  echo "    -r Path of mindspore package to be tested, default is {PROJECT_PATH}/output"
  echo "to be continued ..."
}

checkopts()
{
  DEVICE_TARGET_OPT="ascend310"
  DEVICE_ID_OPT=0
  TASECASE_NAME_OPT=""
  TEST_PATH=${PROJECT_PATH}/tests/st/cpp/cxx_api
  PACKAGE_PATH=${PROJECT_PATH}/output
  PACKAGE_TYPE="cpp"

  # Process the options
  while getopts 'he:d:n:t:r:' opt
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
      t)
        if [[ "X$OPTARG" == "Xcpp" || "X$OPTARG" == "Xpython" ]]; then
          PACKAGE_TYPE="$OPTARG"
        else
          echo "Invalid value ${OPTARG} for option -t"
          usage
          exit 1
        fi
        ;;
      r)
        PACKAGE_PATH=$OPTARG
        echo "package path set to: ${OPTARG}"
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

# using installed or compiled whl packages, set env path by pip
if [[ "${PACKAGE_TYPE}" == "python" ]]; then
    MINDSPORE_PKG_PATH=`python -m pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
  if [[ "X${MINDSPORE_PKG_PATH}" == "X" ]]; then
    MINDSPORE_PKG_PATH=${PROJECT_PATH}/build/package/mindspore:
  fi
elif [[ "${PACKAGE_TYPE}" == "cpp" ]]; then
# using acl tar package, extract tar package here
  rm -rf mindspore_ascend*
  PACKAGE_NAME_FULL=$(find "${PACKAGE_PATH}" -maxdepth 1 -name "mindspore_ascend*.tar.gz")
  PACKAGE_NAME=${PACKAGE_NAME_FULL##*/}

  tar -xzf ${PACKAGE_PATH}/${PACKAGE_NAME}
  MINDSPORE_PKG_PATH=$(find "${TEST_PATH}" -maxdepth 1 -name "mindspore_ascend*")
fi

export LD_LIBRARY_PATH=${MINDSPORE_PKG_PATH}:${MINDSPORE_PKG_PATH}/lib:${PROJECT_PATH}/tests/st/cpp/cxx_api:$LD_LIBRARY_PATH
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
  ./cxx_st_tests --gtest_filter=$TASECASE_NAME_OPT
else
  ./cxx_st_tests
fi
RET=$?
cd -

exit ${RET}
