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
  echo "bash mobilenetv2.sh -r path-to-{mindspore-lite-version-linux-x64.tar.gz}"
  echo "Options:"
  echo "-r : specific path to mindspore-lite-version-linux-x64.tar.gz"
}

nargs=$#
if [ $nargs -eq 0 ] ; then
    usage 
    exit 1 
fi

LITE_PLATFORM="arm64"
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
ROOT_DIR=${BASEPATH%%/mindspore/lite/examples/quick_start_micro/mobilenetv2_arm64}
DEMO_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/mobilenetv2_arm64
MODEL_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/models
PKG_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/pkgs
COFIG_FILE=${DEMO_DIR}/micro.cfg
SOURCE_CODE_DIR=${ROOT_DIR}/mindspore/lite/examples/quick_start_micro/mobilenetv2_arm64/source_code

MODEL_NAME=mobilenetv2
MODEL=${MODEL_DIR}/${MODEL_NAME}/mobilenet_v2_1.0_224.tflite
MODEL_FILE=${MODEL_NAME}.tar.gz

echo "current dir is: ${BASEPATH}"

get_version() {
    VERSION_STR=$(cat ${ROOT_DIR}/version.txt)
}

download_inference() {
    if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
        local ARM_NAME=aarch64
        local DEVICE=gpu
    else
        local ARM_NAME=aarch32
        local DEVICE=cpu
    fi
    rm -rf ${BASEPATH:?}/${MINDSPORE_FILE_NAME} || exit 1
    MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-android-${ARM_NAME}"
    local MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
    local MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/android/${DEVICE}/${MINDSPORE_FILE}"

    if [ ! -e ${PKG_DIR}/${MINDSPORE_FILE} ]; then
      wget -c -O ${PKG_DIR}/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
    fi

    tar xzvf ${PKG_DIR}/${MINDSPORE_FILE} -C ${PKG_DIR} || exit 1
    PKG_PATH=${PKG_DIR}/${MINDSPORE_FILE_NAME}
}

DownloadModel() {
    rm -rf ${MODEL_DIR:?}/${MODEL_NAME}
    mkdir -p ${MODEL_DIR}/${MODEL_NAME}

    local DOWNLOAD_URL=https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/${MODEL_FILE}

    if [ ! -e ${MODEL_DIR}/${MODEL_FILE} ]; then
      echo "download models ..."
      wget -c -O ${MODEL_DIR}/${MODEL_FILE} --no-check-certificate ${DOWNLOAD_URL}
    fi
    echo "unpack models ..."
    tar xzvf ${MODEL_DIR}/${MODEL_FILE} -C ${MODEL_DIR} || exit 1
}

CodeGeneration() {
    tar xzvf ${PKG_DIR}/${MINDSPORE_FILE} -C ${PKG_DIR} || exit 1
    export LD_LIBRARY_PATH=${PKG_DIR}/${MINDSPORE_FILE_NAME}/tools/converter/lib:${LD_LIBRARY_PATH}
    ${PKG_DIR}/${MINDSPORE_FILE_NAME}/tools/converter/converter/converter_lite --fmk=TFLITE --modelFile=${MODEL} --outputFile=${SOURCE_CODE_DIR} --configFile=${COFIG_FILE}
}

get_version
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
download_inference
if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
    echo "making arm64"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI="arm64-v8a" \
          -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang" \
          -DANDROID_NATIVE_API_LEVEL="19" \
          -DPLATFORM_ARM64=ON \
          -DPKG_PATH=${PKG_PATH} ..
else
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI="armeabi-v7a" \
          -DANDROID_TOOLCHAIN_NAME="clang" \
          -DANDROID_NATIVE_API_LEVEL="19" \
          -DPLATFORM_ARM32=ON \
          -DPKG_PATH=${PKG_PATH} ..
fi
make
