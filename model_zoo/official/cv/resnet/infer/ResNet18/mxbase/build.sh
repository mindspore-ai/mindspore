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

path_cur=$(dirname "$0")

function check_env()
{
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
    if [ ! "${ASCEND_VERSION}" ]; then
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}

function build_resnet18()
{
    cd "$path_cur" || exit
    rm -rf build
    mkdir -p build
    cd build || exit
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build resnet18."
        exit ${ret}
    fi
    make install
}

check_env
build_resnet18
