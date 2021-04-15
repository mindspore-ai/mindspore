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
  echo "bash build.sh [-I arm64|arm32]"
  echo "Options:"
  echo "    -I download and build for arm64 or arm32, default arm64"
}

LITE_PLATFORM="arm64"
while getopts 'I:' OPT
do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case $OPT in
        I)
            if [[ "$OPTARG" == "arm64" ]]; then
              LITE_PLATFORM="arm64"
            elif [[ "$OPTARG" == "arm32" ]]; then
              LITE_PLATFORM="arm32"
            else
              echo "-I parameter must be arm64 or arm32"
              exit 1
            fi
            ;;
        *)
            echo "Unknown option ${opt}!"
            usage
            exit 1
    esac
done

BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MINDSPORE_ROOT_DIR=${BASEPATH%%/mindspore/lite/micro/example/mobilenetv2}

echo "current dir is: ${BASEPATH}"

MOBILE_NAME=mobilenetv2
MOBILE_FILE=${MOBILE_NAME}.ms

get_version() {
    local VERSION_HEADER=${MINDSPORE_ROOT_DIR}/mindspore/lite/include/version.h
    local VERSION_MAJOR=$(grep "const int ms_version_major =" ${VERSION_HEADER} | tr -dc "[0-9]")
    local VERSION_MINOR=$(grep "const int ms_version_minor =" ${VERSION_HEADER} | tr -dc "[0-9]")
    local VERSION_REVISION=$(grep "const int ms_version_revision =" ${VERSION_HEADER} | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}

download_inference() {
    if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
        local ARM_NAME=aarch64
    else
        local ARM_NAME=aarch32
    fi
    MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-android-${ARM_NAME}"
    local MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
    local MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/android/${MINDSPORE_FILE}"

    if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
      wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
    fi

    tar xzvf ${BASEPATH}/build/${MINDSPORE_FILE} -C ${BASEPATH}/build/ || exit 1
    rm ${BASEPATH}/build/${MINDSPORE_FILE} || exit 1
    PKG_PATH=${BASEPATH}/build/${MINDSPORE_FILE_NAME}
}

download_mobile() {
    local MOBILE_DOWNLOAD_URL=https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_imagenet/r1.2/${MOBILE_FILE}

    if [ ! -e ${BASEPATH}/build/${MOBILE_FILE} ]; then
      wget -c -O ${BASEPATH}/build/${MOBILE_FILE} --no-check-certificate ${MOBILE_DOWNLOAD_URL}
    fi
}

gen_mobile() {
    local CODEGEN_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
    local CODEGEN_FILE="${CODEGEN_FILE_NAME}.tar.gz"
    local CODEGEN_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/${CODEGEN_FILE}"

    if [ ! -e ${BASEPATH}/build/${CODEGEN_FILE} ]; then
      wget -c -O ${BASEPATH}/build/${CODEGEN_FILE} --no-check-certificate ${CODEGEN_LITE_DOWNLOAD_URL}
    fi

    tar xzvf ${BASEPATH}/build/${CODEGEN_FILE} -C ${BASEPATH}/build/ || exit 1
    rm ${BASEPATH}/build/${CODEGEN_FILE} || exit 1
    CODEGEN_PATH=${BASEPATH}/build/${CODEGEN_FILE_NAME}/tools/codegen
    if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
        local TARGET=ARM64
    else
        local TARGET=ARM32A
    fi
    ${CODEGEN_PATH}/codegen --codePath=${BASEPATH}/build --modelPath=${BASEPATH}/build/${MOBILE_FILE} --target=${TARGET}
}

mkdir -p ${BASEPATH}/build

get_version
download_inference

echo "downloading ${MOBILE_FILE}!"
download_mobile
echo "generating mobilenetv2"
gen_mobile
BENCHMARK_PATH=${BASEPATH}/build/${MOBILE_NAME}

# build benchmark
rm -rf ${BASEPATH}/build/benchmark
mkdir -p ${BASEPATH}/build/benchmark && cd ${BASEPATH}/build/benchmark || exit 1

if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
    echo "making arm64"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI="arm64-v8a" \
          -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang" \
          -DANDROID_NATIVE_API_LEVEL="19" \
          -DPLATFORM_ARM64=ON \
          -DPKG_PATH=${PKG_PATH} ${BENCHMARK_PATH}
else
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI="armeabi-v7a" \
          -DANDROID_TOOLCHAIN_NAME="clang" \
          -DANDROID_NATIVE_API_LEVEL="19" \
          -DPLATFORM_ARM32=ON \
          -DPKG_PATH=${PKG_PATH} ${BENCHMARK_PATH}
fi
make
