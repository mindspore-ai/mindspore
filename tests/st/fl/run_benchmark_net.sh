#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

usage()
{
  echo "Usage:"
  echo "bash run_benchmark_net.sh [-h] [-l level] [-r PATH] [-p PATH] [-j PATH] [-m PATH]"
  echo "Options:"
  echo "    -h Print usage"
  echo "    -l st level, now not used"
  echo "    -r fl resource path"
  echo "    -p mindspore x86_whl and lite_x86 pkg/"
  echo "    -j java jdk path, version 1.9 or later"
  echo "    -m example module path"
}

base_path=$(dirname "$(readlink -f "$0")")
# Process the options
while getopts 'h:l:r:p:j:m:' opt
do
  case "${opt}" in
    h)
      usage;;
    l)
      FL_TEST_LEVEL=${OPTARG}
      echo "FL_TEST_LEVEL is ${FL_TEST_LEVEL}";;
    r)
      FL_RESOURCE_PATH=${OPTARG}
      echo "FL_RESOURCE_PATH is ${FL_RESOURCE_PATH}";;
    p)
      X86_PKG_PATH=${OPTARG}
      echo "X86_PKG_PATH is ${X86_PKG_PATH}";;
    j)
      FL_JDK_PATH=${OPTARG}
      echo "FL_JDK_PATH is ${FL_JDK_PATH}";;
    m)
      FL_MODELS_PATH=${OPTARG}
      echo "FL_MODELS_PATH is ${FL_MODELS_PATH}";;
    *)
      echo "Unknown option ${opt}!"
      usage
      exit 1
  esac
done

if [ "X${FL_JDK_PATH}" == "X" ] && [ "X${JAVA_HOME}" != "X" ]; then
  FL_JDK_PATH=${JAVA_HOME}/bin
fi

if [ "X${FL_RESOURCE_PATH}"  == "X" ]; then
  FL_RESOURCE_PATH=${base_path}/fl_resources
fi

if [ "X${FL_RESOURCE_PATH}" == "X" ] || [ "X${X86_PKG_PATH}" == "X" ] ||  \
  [ "X${FL_JDK_PATH}" == "X" ] || [ "X${FL_MODELS_PATH}" == "X" ]; then
  echo "FL_RESOURCE_PATH, X86_PKG_PATH, FL_JDK_PATH must be set."
  usage
  exit 1
fi

cd $base_path/mobile/ci_script

echo "lenet_train_and_infer.sh ${FL_RESOURCE_PATH} ${X86_PKG_PATH} ${FL_JDK_PATH} ${FL_MODELS_PATH} start..."
bash lenet_train_and_infer.sh ${FL_RESOURCE_PATH} ${X86_PKG_PATH} ${FL_JDK_PATH} ${FL_MODELS_PATH}
ret=$?
echo "lenet_train_and_infer.sh ${FL_RESOURCE_PATH} ${X86_PKG_PATH} ${FL_JDK_PATH} ${FL_MODELS_PATH}  success."
exit $ret
