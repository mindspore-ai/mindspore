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

mkdir -p build

if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi

tar xzvf ${BASEPATH}/build/${MINDSPORE_FILE} -C ${BASEPATH}/build/ || exit 1
rm ${BASEPATH}/build/${MINDSPORE_FILE} || exit 1
CODEGEN_PATH=${BASEPATH}/build/${MINDSPORE_FILE_NAME}/tools/codegen
HEADER_PATH=${BASEPATH}/build/${MINDSPORE_FILE_NAME}/inference
# 1. build static lib.a
echo -e "building static library"
mkdir -p ${BASEPATH}/build/src && cd ${BASEPATH}/build/src || exit 1
OP_HEADER_PATH=${CODEGEN_PATH}/operator_library/include
OP_LIB=${CODEGEN_PATH}/operator_library/lib/libops.a
echo "Head Path: ${OP_HEADER_PATH}"
echo "Lib Path: ${OP_LIB}"
echo "Header Path: ${HEADER_PATH}"

cmake -DCMAKE_BUILD_TYPE=Debug            \
      -DOP_LIB=${OP_LIB}                  \
      -DOP_HEADER_PATH=${OP_HEADER_PATH}  \
      -DHEADER_PATH=${HEADER_PATH}        \
      ${BASEPATH}/src
make

# 2. build benchmark
mkdir -p ${BASEPATH}/build/benchmark && cd ${BASEPATH}/build/benchmark || exit 1
cmake -DMODEL_LIB="${BASEPATH}/build/src/libnet.a"  \
      -DHEADER_PATH=${HEADER_PATH}                  \
      ${BASEPATH}/benchmark
make

echo "net file: ${BASEPATH}/src/mnist.net"
# 3. run benchmark
./benchmark ${INPUT_BIN} ${BASEPATH}/src/net.net
