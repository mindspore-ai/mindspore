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

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
get_version() {
    VERSION_STR=$(cat ${BASEPATH}/../../../../../version.txt)
}
get_version
MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir"
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/x86_64/${MINDSPORE_FILE}"

mkdir -p build
mkdir -p lib/runtime
mkdir -p lib/tools/converter
mkdir -p model
if [ ! -e ${BASEPATH}/model/mobilenetv2.mindir ]; then
    wget -c -O ${BASEPATH}/model/mobilenetv2.mindir --no-check-certificate ${MODEL_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
  wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi
tar xzvf ${BASEPATH}/build/${MINDSPORE_FILE} -C ${BASEPATH}/build/

# copy shared libraries to lib
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/runtime/lib/ ${BASEPATH}/lib/runtime/
cd ${BASEPATH}/ || exit

mvn package
