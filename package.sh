#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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

BASEPATH=$(cd "$(dirname $0)"; pwd)
echo "${BASEPATH}"
cd "${BASEPATH}"
BUILD_PATH="${BASEPATH}/build"
PACKAGE_PATH="${BUILD_PATH}/package"
OUTPUT_PATH="${BASEPATH}/output"

mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

to_lower () {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

COMMIT_ID=$(git log --format='[sha1]:%h,[branch]:%d' -1 | sed 's/ //g')
export COMMIT_ID

PYTHON=$(which python3)
PYTHON_VERSION=$("${PYTHON}" -V 2>&1 | awk '{print $2}' | cut -d. -f-2)
if [[ $(uname) == "Linux" ]]; then
    if [[ "${PYTHON_VERSION}" == "3.7" ]]; then
        PY_TAGS="cp37-cp37m"
    else
        echo "Could not find 'Python 3.7'"
        exit 1
    fi
    PLATFORM_TAG=$(to_lower "$(uname)_$(uname -m)")
elif [[ $(uname) == "Darwin" ]]; then
    if [[ "${PYTHON_VERSION}" == "3.7" ]]; then
        PY_TAGS="py3-none"
    else
        echo "Could not find 'Python 3.7'"
        exit 1
    fi
    PLATFORM_TAG="any"
fi
echo "=========${BASEPATH}==================="
mk_new_dir "${OUTPUT_PATH}"

#copy necessary file to pack_path
cp ${BASEPATH}/mindspore/*.py "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/nn" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/_extends" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/parallel" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/mindrecord" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/train" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/model_zoo" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/common" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/ops" "${PACKAGE_PATH}/mindspore"
cp -rf "${BUILD_PATH}/../mindspore/communication" "${PACKAGE_PATH}/mindspore"

if [[ "X$2" = "Xgpu" ]]; then
    echo "package _akg when gpu enable."
    cp -rf "${BASEPATH}/mindspore/_akg" "${PACKAGE_PATH}"
    if [[ -d "${BUILD_PATH}/mindspore/incubator-tvm" ]]; then
        cp -rf "${BUILD_PATH}/mindspore/incubator-tvm/topi/python/topi" "${PACKAGE_PATH}/_akg"
        cp -rf "${BUILD_PATH}/mindspore/incubator-tvm/python/tvm" "${PACKAGE_PATH}/_akg"
    fi
fi

# move dataset
if [[ -d "${BASEPATH}/mindspore/dataset" ]]; then
    cp -rf "${BASEPATH}/mindspore/dataset" "${PACKAGE_PATH}/mindspore"
fi

cd "${PACKAGE_PATH}"
if [ -n "$1" ];then
    export BACKEND_POLICY=$1
else
    export BACKEND_POLICY="ms"
fi

# package name
if [[ "X$1" = "Xge" ]]; then
    export MS_PACKAGE_NAME="mindspore"
elif [[ "X$1" = "Xms" && "X$2" = "Xgpu" ]]; then
    export MS_PACKAGE_NAME="mindspore-gpu"
elif [[ "X$1" = "Xms" && "X$2" = "Xascend" ]]; then
    export MS_PACKAGE_NAME="mindspore-ascend"
elif [[ "X$1" = "Xms" && "X$2" = "Xcpu" ]]; then
    export MS_PACKAGE_NAME="mindspore"
else
    export MS_PACKAGE_NAME="mindspore"
fi

${PYTHON} "${BASEPATH}/setup.py" bdist_wheel

chmod -R 700 ${PACKAGE_PATH}/mindspore/
chmod -R 700 ${PACKAGE_PATH}/${MS_PACKAGE_NAME//-/_}.egg-info/

# rename package
PACKAGE_FULL_NAME=$(find "${PACKAGE_PATH}" -iname "*.whl")
PACKAGE_BASE_NAME=$(echo ${PACKAGE_FULL_NAME} | awk -F / '{print $NF}' | awk -F - '{print $1"-"$2}')
PACKAGE_BASE_NAME=${PACKAGE_BASE_NAME//_*-/-}

PACKAGE_NEW_NAME="${PACKAGE_BASE_NAME}-${PY_TAGS}-${PLATFORM_TAG}.whl"
cp -rf "${PACKAGE_PATH}/dist"/*.whl "${PACKAGE_PATH}/${PACKAGE_NEW_NAME}"
cp -f "${PACKAGE_PATH}/${PACKAGE_NEW_NAME}" "${OUTPUT_PATH}"
find ${OUTPUT_PATH} -name "*.whl" -print0 | xargs -0 -I {} sh -c "sha256sum {} | awk '{printf \$1}' > {}.sha256"

cd "${BASEPATH}"

echo "------Successfully created mindspore package------"
