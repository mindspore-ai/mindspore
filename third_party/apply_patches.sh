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

PWD_PATH=`pwd`
THIRD_PARTY_PATH=$(cd "$(dirname $0)"; pwd)
if [ $# -lt 1 ]; then
    echo "Usage: sh apply_patches.sh [build_dir]"
    echo "       build_dir is the directory where you type \"cmake\""
    echo "       Open source software incubator-tvm will be copied to build_dir"
    echo "           where patches will be applied on."
    exit 1
fi
BUILD_PATH=$1

if [ -d ${BUILD_PATH}/incubator-tvm ]; then
    rm -rf ${BUILD_PATH}/incubator-tvm
fi
DLPACK_PATH=$2
DMLC_PATH=$3
RANG_PATH=$4
TVM_PATH=$5
mkdir ${BUILD_PATH}/incubator-tvm
cp  -rf ${TVM_PATH}/* ${BUILD_PATH}/incubator-tvm/
cp  -rf ${DLPACK_PATH}/* ${BUILD_PATH}/incubator-tvm/3rdparty/dlpack/
cp  -rf ${DMLC_PATH}/* ${BUILD_PATH}/incubator-tvm/3rdparty/dmlc-core/
cp  -rf ${RANG_PATH}/* ${BUILD_PATH}/incubator-tvm/3rdparty/rang/


check_dir_not_empty()
{
    if [ ! $# -eq 1 ]; then
        echo "Usage: check_dir_not_empty dir_path"
	exit 1
    fi

    if [ ! -d $1 ]; then
        echo "Directory $1 does not exist."
	exit 1
    fi

    fileCounts=`ls $1 | wc -l`
    if [ ${fileCounts} -eq 0 ]; then
        echo "Directory $1 is empty."
	exit 1
    fi
}

apply_patch()
{
    if [ ! $# -eq 1 ]; then
        echo "Usage: apply_patch patch_name"
	exit 1
    fi

    if [ ! -f $1 ]; then
        echo "Patch $1 does not exist."
        exit 1
    fi

    patch -p1 < $1
    if [ $? -eq 0 ]; then
        echo "Patch $1 applied successfully."
    else
        echo "Patch $1 not applied."
    fi
}

# apply patches on tvm
TVM_PATH=${BUILD_PATH}/incubator-tvm
TVM_PATCH_PATH=${THIRD_PARTY_PATH}/patch/incubator-tvm
check_dir_not_empty "${TVM_PATH}"
check_dir_not_empty "${TVM_PATCH_PATH}"
cd ${TVM_PATH}
apply_patch "${TVM_PATCH_PATH}/cmake.patch"
apply_patch "${TVM_PATCH_PATH}/find_library.patch"
apply_patch "${TVM_PATCH_PATH}/include.patch"
apply_patch "${TVM_PATCH_PATH}/src_pass.patch"

cd ${PWD_PATH}
