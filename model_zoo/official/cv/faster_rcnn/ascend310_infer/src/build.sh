#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
path_cur=$(cd "`dirname $0`" || exit; pwd)
build_type="Release"

function preparePath() {
    rm -rf $1
    mkdir -p $1
    cd $1
}

function buildA300() {
    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to acllib when it was not specified by user
        export ARCH_PATTERN=acllib
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user, reset it to ${ARCH_PATTERN}/acllib"
        export ARCH_PATTERN=${ARCH_PATTERN}/acllib
    fi

    path_build=$path_cur/build
    preparePath $path_build
    cmake -DCMAKE_BUILD_TYPE=$build_type ..
    make -j
    ret=$?
    cd ..
    return ${ret}
}

# set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
if [ ! "${ASCEND_VERSION}" ]; then
    export ASCEND_VERSION=ascend-toolkit/latest
    echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
else
    echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
fi

buildA300

if [ $? -ne 0 ]; then
    exit 1
fi