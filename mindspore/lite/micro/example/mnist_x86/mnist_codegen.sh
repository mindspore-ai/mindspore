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

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MINDSPORE_ROOT_DIR=${${BASEPATH}%%/mindspore/lite/micro/example/mnist}

echo "current dir is: ${BASEPATH}"

VERSION_HEADER=${MINDSPORE_ROOT_DIR}/mindspore/lite/include/version.h
INPUT_BIN=${BASEPATH}/mnist_input.bin

get_version() {
    VERSION_MAJOR=$(grep "const int ms_version_major =" ${VERSION_HEADER} | tr -dc "[0-9]")
    VERSION_MINOR=$(grep "const int ms_version_minor =" ${VERSION_HEADER} | tr -dc "[0-9]")
    VERSION_REVISION=$(grep "const int ms_version_revision =" ${VERSION_HEADER} | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}
get_version
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-inference-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/${MINDSPORE_FILE}"

MNIST_NAME=mnist
MNIST_FILE=${MNIST_NAME}.ms
MNIST_DOWNLOAD_URL=https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/${MNIST_FILE}

mkdir -p build

if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi

if [ ! -e ${BASEPATH}/build/${MNIST_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${MNIST_FILE} --no-check-certificate ${MNIST_DOWNLOAD_URL}
fi

tar xzvf ${BASEPATH}/build/${MINDSPORE_FILE} -C ${BASEPATH}/build/ || exit 1
rm ${BASEPATH}/build/${MINDSPORE_FILE} || exit 1
PKG_PATH=${BASEPATH}/build/${MINDSPORE_FILE_NAME}

# 1. codegen
${BASEPATH}/build/${MINDSPORE_FILE_NAME}/tools/codegen/codegen --codePath=${BASEPATH}/build --modelPath=${BASEPATH}/build/${MNIST_FILE}

# 2. build benchmark
mkdir -p ${BASEPATH}/build/benchmark && cd ${BASEPATH}/build/benchmark || exit 1
cmake -DPKG_PATH=${PKG_PATH} ${BASEPATH}/build/${MNIST_NAME}
make

# 3. run benchmark
echo "net file: ${BASEPATH}/src/mnist.bin"
./benchmark ${INPUT_BIN} ${BASEPATH}/src/net.bin
