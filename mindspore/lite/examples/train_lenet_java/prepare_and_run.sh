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

#!/bin/bash

display_usage()
{
  echo -e "\nUsage: prepare_and_run.sh -D dataset_path [-d mindspore_docker] [-r release.tar.gz] [-m mindir]\n"
}

checkopts()
{
  DOCKER=""
  MINDIR_FILE=""
  MNIST_DATA_PATH=""
  while getopts 'D:d:m:r:' opt
  do
    case "${opt}" in
      D)
        MNIST_DATA_PATH=$(realpath $OPTARG)
        ;;
      d)
        DOCKER=$OPTARG
        ;;
      m)
        MINDIR_FILE=$OPTARG
        ;;
      r)
        TARBALL="-r $OPTARG"
        ;;
      *)
        echo "Unknown option ${opt}!"
        display_usage
        exit 1
    esac
  done
}

START=$(date +%s.%N)
checkopts "$@"
if [ "$MNIST_DATA_PATH" == "" ] || [ ! -d "$MNIST_DATA_PATH" ]; then
  echo "MNIST Dataset directory path was not provided or wrong path " $MNIST_DATA_PATH
  display_usage
  exit 1
fi

./build.sh $TARBALL

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)

EXPORT=""
if [ "$MINDIR_FILE" != "" ]; then
  cp -f $MINDIR_FILE model/lenet_tod.mindir
  EXPORT="DONT_EXPORT"
fi

cd model/ || exit 1
MSLITE_LINUX=$(ls -d ${BASEPATH}/build/mindspore-lite-*-linux-x64)
CONVERTER=${MSLITE_LINUX}/tools/converter/converter/converter_lite
rm -f *.ms
LD_LIBRARY_PATH=${MSLITE_LINUX}/tools/converter/lib/:${MSLITE_LINUX}/tools/converter/third_party/glog/lib
EXPORT=${EXPORT} LD_LIBRARY_PATH=${LD_LIBRARY_PATH} CONVERTER=${CONVERTER} ./prepare_model.sh $DOCKER || exit 1
cd ../

if [ "$TARBALL" != "" ]; then
  rm -rf build
fi

cd target || exit 1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib/ 
java -Djava.library.path=../lib/ -classpath .:./train_lenet_java.jar:../lib/mindspore-lite-java.jar  com.mindspore.lite.train_lenet.Main ../model/lenet_tod.ms $MNIST_DATA_PATH 1
cd -
END=$(date +%s.%N)
TIME=$(echo "$END-$START" | bc)
echo "total run train lenet java time: $TIME s"
