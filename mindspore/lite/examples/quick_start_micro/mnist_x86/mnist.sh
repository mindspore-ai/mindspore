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

GEN=OFF
TARBALL=""
while getopts 'r:g:' OPT
do
    case "${OPT}" in
        g)
            GEN=$OPTARG
            ;;
        r)
            TARBALL=$OPTARG
            ;;
        ?)
            echo "Usage: add -g on , -r specific release.tar.gz"
    esac
done

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=${BASEPATH%%/mindspore/lite/examples/quick_start_micro/mnist_x86}
DEMO_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/mnist_x86
MODEL_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/models
PKG_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/pkgs
COFIG_FILE=${DEMO_DIR}/micro.cfg
echo "root dir is: ${ROOT_DIR}"
echo "current dir is: ${BASEPATH}"
echo "demo dir is: ${DEMO_DIR}"
echo "model dir is: ${MODEL_DIR}"

MODEL_NAME=mnist
INPUT_BIN=${MODEL_DIR}/${MODEL_NAME}/mnist.tflite.ms.bin
VALICATION_DATA=${MODEL_DIR}/${MODEL_NAME}/mnist.tflite.ms.out
MODEL=${MODEL_DIR}/${MODEL_NAME}/mnist.tflite
MODEL_FILE=${MODEL_NAME}.tar.gz

get_version() {
    local VERSION_HEADER=${ROOT_DIR}/mindspore/lite/include/version.h
    local VERSION_MAJOR=$(grep "const int ms_version_major =" ${VERSION_HEADER} | tr -dc "[0-9]")
    local VERSION_MINOR=$(grep "const int ms_version_minor =" ${VERSION_HEADER} | tr -dc "[0-9]")
    local VERSION_REVISION=$(grep "const int ms_version_revision =" ${VERSION_HEADER} | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}

download_mnist() {
    rm -rf ${MODEL_DIR:?}/${MODEL_NAME}
    mkdir -p ${MODEL_DIR}/${MODEL_NAME}
    tar xzvf ${MODEL_DIR}/${MODEL_FILE} -C ${MODEL_DIR}/${MODEL_NAME} || exit 1
}

gen_mnist() {
    tar xzvf ${PKG_DIR}/${MINDSPORE_FILE} -C ${PKG_DIR} || exit 1
    export LD_LIBRARY_PATH=${PKG_DIR}/${MINDSPORE_FILE_NAME}/tools/converter/lib:${LD_LIBRARY_PATH}
    ${PKG_DIR}/${MINDSPORE_FILE_NAME}/tools/converter/converter/converter_lite --fmk=TFLITE --modelFile=${MODEL} --outputFile=${DEMO_DIR} --configFile=${COFIG_FILE}
}

mkdir -p ${BASEPATH}/build

get_version
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
PKG_PATH=${PKG_DIR}/${MINDSPORE_FILE_NAME}

echo "tar ball is: ${TARBALL}"
if [ -n "$TARBALL" ]; then
   echo "cp file"
   rm -rf ${PKG_DIR}
   mkdir -p ${PKG_DIR}
   cp ${TARBALL} ${PKG_DIR}
fi

# 1. code-generation
if [[ "${GEN}" == "ON" ]] || [[ "${GEN}" == "on" ]]; then
    echo "downloading mnist.ms!"
    download_mnist
    echo "generating mnist"
    gen_mnist
fi

# 2. build benchmark
mkdir -p ${DEMO_DIR}/build && cd ${DEMO_DIR}/build || exit 1
cmake -DPKG_PATH=${PKG_PATH} ..
make

# 3. run benchmark
echo "net file: ${DEMO_DIR}/src/mnist.bin"
./benchmark ${INPUT_BIN} ../src/net.bin 1 ${VALICATION_DATA}