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

usage()
{
  echo "Usage:"
  echo "bash mnist.sh -r path-to-{mindspore-lite-version-linux-x64.tar.gz}"
  echo "Options:"
  echo "-r : specific path to mindspore-lite-version-linux-x64.tar.gz"
}

nargs=$#
if [ $nargs -eq 0 ] ; then
    usage
    exit 1
fi

while getopts 'r:' OPT
do
    case "${OPT}" in
        r)
            TARBALL=$OPTARG
            ;;
        ?)
            echo "Usage: -r specific release.tar.gz"
    esac
done


BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=${BASEPATH%%/mindspore/lite/examples/quick_start_micro/mnist_x86}
DEMO_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/mnist_x86
MODEL_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/models
PKG_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/pkgs
COFIG_FILE=${DEMO_DIR}/micro.cfg
SOURCE_CODE_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/mnist_x86/source_code


MODEL_NAME=mnist
INPUT_BIN=${MODEL_DIR}/${MODEL_NAME}/mnist.tflite.ms.bin
VALICATION_DATA=${MODEL_DIR}/${MODEL_NAME}/mnist.tflite.ms.out
MODEL=${MODEL_DIR}/${MODEL_NAME}/mnist.tflite
MODEL_FILE=${MODEL_NAME}.tar.gz

GetVersion() {
    VERSION_STR=$(cat ${ROOT_DIR}/version.txt)
}

DownloadModel() {
    rm -rf ${MODEL_DIR:?}/${MODEL_NAME}
    mkdir -p ${MODEL_DIR}/${MODEL_NAME}

    local MNIST_DOWNLOAD_URL=https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/${MODEL_FILE}

    if [ ! -e ${MODEL_DIR}/${MODEL_FILE} ]; then
      echo "download models ..."
      wget -c -O ${MODEL_DIR}/${MODEL_FILE} --no-check-certificate ${MNIST_DOWNLOAD_URL}
    fi
    echo "unpack models ..."
    tar xzvf ${MODEL_DIR}/${MODEL_FILE} -C ${MODEL_DIR} || exit 1
}

CodeGeneration() {
    tar xzvf ${PKG_DIR}/${MINDSPORE_FILE} -C ${PKG_DIR} || exit 1
    export LD_LIBRARY_PATH=${PKG_DIR}/${MINDSPORE_FILE_NAME}/tools/converter/lib:${LD_LIBRARY_PATH}
    ${PKG_DIR}/${MINDSPORE_FILE_NAME}/tools/converter/converter/converter_lite --fmk=TFLITE --modelFile=${MODEL} --outputFile=${SOURCE_CODE_DIR} --configFile=${COFIG_FILE}
}

GetVersion
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
PKG_PATH=${PKG_DIR}/${MINDSPORE_FILE_NAME}

echo "tar ball is: ${TARBALL}"
if [ -n "$TARBALL" ]; then
   echo "cp file"
   rm -rf ${PKG_PATH}
   mkdir -p ${PKG_PATH}
   cp ${TARBALL} ${PKG_DIR}
fi

# 1. code-generation
echo "downloading ${MODEL_FILE}!"
DownloadModel
echo "micro code-generation"
CodeGeneration

# 2. build benchmark
mkdir -p ${SOURCE_CODE_DIR}/build && cd ${SOURCE_CODE_DIR}/build || exit 1
cmake -DPKG_PATH=${PKG_PATH} ..
make

# 3. run benchmark
echo "net file: ${DEMO_DIR}/src/${MODEL_NAME}.bin"
./benchmark ${INPUT_BIN} ../src/net.bin 1 ${VALICATION_DATA}

