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
BASEPATH=$(
  cd "$(dirname "$0")"
  pwd
)
PROJECT_PATH=${BASEPATH}/../../..
if [ $BUILD_PATH ]; then
  echo "BUILD_PATH = $BUILD_PATH"
else
  BUILD_PATH=${PROJECT_PATH}/build
  echo "BUILD_PATH = $BUILD_PATH"
fi
cd ${BUILD_PATH}/mindspore/tests/ut/cpp

export LD_LIBRARY_PATH=${BUILD_PATH}/mindspore/googletest/googlemock/gtest:${PROJECT_PATH}/mindspore/python/mindspore:${PROJECT_PATH}/mindspore/python/mindspore/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT_PATH}/tests/ut/cpp/python_input:$PYTHONPATH:${PROJECT_PATH}/mindspore/python:${PROJECT_PATH}/tests/ut/python:${PROJECT_PATH}
export GLOG_v=2

## prepare data for dataset & mindrecord
cp -fr $PROJECT_PATH/tests/ut/data ${PROJECT_PATH}/build/mindspore/tests/ut/cpp/
## prepare album dataset, uses absolute path so has to be generated
python ${PROJECT_PATH}/build/mindspore/tests/ut/cpp/data/dataset/testAlbum/gen_json.py

## prepare data for graph kernel
cp -fr $PROJECT_PATH/tests/ut/cpp/backend/ms_backend/graph_fusion/opt/config ${PROJECT_PATH}/build/mindspore/tests/ut/cpp/graph_kernel_config

RET=0
if [ $# -gt 0 ]; then
  ./ut_CORE_tests --gtest_filter=$1
  ./ut_FRONTEND_tests --gtest_filter=$1
  ./ut_OLD_BACKEND_tests --gtest_filter=$1
  ./ut_BACKEND_tests --gtest_filter=$1
  ./ut_PS_tests --gtest_filter=$1
  ./ut_OTHERS_tests --gtest_filter=$1
  ./ut_MINDDATA0_tests --gtest_filter=$1
  ./ut_MINDDATA1_tests --gtest_filter=$1
  ./ut_CCSRC_tests --gtest_filter=$1
  ./ut_SYMBOL_ENGINE_tests --gtest_filter=$1
  ./ut_GRAPH_KERNEL_tests --gtest_filter=$1
  ./ut_INTERNAL_KERNEL_tests --gtest_filter=$1
  ./ut_PYNATIVE_tests --gtest_filter=$1
  ./ut_ERROR_HANDLE_tests --gtest_filter=$1
  exit 0
fi

set +e

#pids=()
tasks=(./ut_CORE_tests ./ut_ERROR_HANDLE_tests ./ut_FRONTEND_tests ./ut_BACKEND_tests ./ut_PS_tests ./ut_OTHERS_tests ./ut_MINDDATA0_tests ./ut_MINDDATA1_tests ./ut_CCSRC_tests ./ut_SYMBOL_ENGINE_tests ./ut_GRAPH_KERNEL_tests ./ut_INTERNAL_KERNEL_tests ./ut_PYNATIVE_tests)

set +e
for task in "${tasks[@]}"; do
  $task
#  $task &
#  pids+=($!)
  status=$?
  if [ $status != 0 ]; then
    RET=$status
    exit $RET
  fi
done
cd -
#for pid in "${pids[@]}"; do
#  wait $pid
#  status=$?
#  if [ $status != 0 ]; then
#    RET=$status
#  fi
#done

exit $RET
