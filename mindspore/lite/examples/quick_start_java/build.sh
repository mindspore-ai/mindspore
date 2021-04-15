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

BASEPATH=$(cd "$(dirname $0)"; pwd)
get_version() {
    VERSION_MAJOR=$(grep "const int ms_version_major =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
    VERSION_MINOR=$(grep "const int ms_version_minor =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
    VERSION_REVISION=$(grep "const int ms_version_revision =" ${BASEPATH}/../../include/version.h | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}
get_version
MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_imagenet/mobilenetv2.ms"
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/${MINDSPORE_FILE}"

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
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/inference/lib/jar/* ${BASEPATH}/lib
cd ${BASEPATH}/

mvn package
