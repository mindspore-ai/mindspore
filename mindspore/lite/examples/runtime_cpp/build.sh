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

usage()
{
  echo "Usage:"
  echo "bash build.sh [-e npu] "
  echo ""
  echo "Options:"
  echo "    If set to -e npu, we will download the library of CPU+NPU, otherwise it will download the library of CPU+GPU by default."
}

BASEPATH=$(
  cd "$(dirname $0)"
  pwd
)
get_version() {
  VERSION_MAJOR=$(grep "const int ms_version_major =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
  VERSION_MINOR=$(grep "const int ms_version_minor =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
  VERSION_REVISION=$(grep "const int ms_version_revision =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
  VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}

# check and set options
checkopts()
{
  # Init default values of build options
  DEVICE="GPU"
  SUPPORT_NPU="off"
  MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-android-aarch64"
  MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
  MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/android/gpu/${MINDSPORE_FILE}"
  # Process the options
  while getopts 'e:h' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      e)
        DEVICE=$OPTARG
        if [[ "X${DEVICE}" == "Xgpu" ]]; then
          continue
        elif [[ "X${DEVICE}" == "Xnpu" ]]; then
          MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-android-aarch64"
          MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/android/${MINDSPORE_FILE}"
          SUPPORT_NPU="on"
        else
          echo "Unknown DEVICE option ${OPTARG}!"
          usage
          exit 1
        fi
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

get_version
checkopts "$@"

MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_imagenet/mobilenetv2.ms"

mkdir -p build
mkdir -p lib
mkdir -p model
if [ ! -e ${BASEPATH}/model/mobilenetv2.ms ]; then
  wget -c -O ${BASEPATH}/model/mobilenetv2.ms --no-check-certificate ${MODEL_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi
tar xzvf ${BASEPATH}/build/${MINDSPORE_FILE} -C ${BASEPATH}/build/
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/inference/lib/libmindspore-lite.a ${BASEPATH}/lib
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/inference/include ${BASEPATH}/
if [[ "X${DEVICE}" == "Xnpu" ]]; then
    cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/inference/third_party/hiai_ddk/lib/*.so ${BASEPATH}/lib
fi
cd ${BASEPATH}/build
cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19" \
  -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_STL="c++_shared" ${BASEPATH} -DSUPPORT_NPU=${SUPPORT_NPU}

make && make install && make package
