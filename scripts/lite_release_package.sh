#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

function android_release_package()
{
    arch=$1
    device=$2
    pkg_name="mindspore-lite-${version}-android-${arch}"

    rm -rf ${pkg_name}
    tar -xzf ${input_path}/android_${arch}/${device}/${pkg_name}.tar.gz
    # Copy java runtime to Android package
    cp ${input_path}/aar/mindspore-lite-*maven*.zip ${pkg_name}

    mkdir -p ${output_path}/release/android/${device}/
    tar -czf ${output_path}/release/android/${device}/${pkg_name}.tar.gz ${pkg_name}
    rm -rf ${pkg_name}
    cd ${output_path}/release/android/${device}/
    sha256sum ${pkg_name}.tar.gz > ${pkg_name}.tar.gz.sha256
}

function ios_release_package()
{
    mkdir -p ${output_path}/release/ios/
    cp ${input_path}/ios_aarch64/*.tar.gz* ${output_path}/release/ios/
    cp ${input_path}/ios_aarch32/*.tar.gz* ${output_path}/release/ios/
}

function linux_release_package()
{
    mkdir -p ${output_path}/release/linux/nnie/
    cp ${input_path}/ubuntu_x86/avx/*.tar.gz* ${output_path}/release/linux/

    cp ${input_path}/linux_aarch32/*.tar.gz* ${output_path}/release/linux/
    cp ${input_path}/ubuntu_x86/nnie/3516D/*.tar.gz* ${output_path}/release/linux/nnie/
}

function windows_release_package()
{
    mkdir -p ${output_path}/release/windows/
    cp ${input_path}/windows_x64/avx/*.zip* ${output_path}/release/windows/
    cp ${input_path}/windows_x32/sse/*.zip* ${output_path}/release/windows/
}

function openharmony_release_package()
{
    mkdir -p ${output_path}/release/openharmony/
    cp ${input_path}/ohos_aarch32/*.tar.gz* ${output_path}/release/openharmony/
}

echo "============================== begin =============================="
echo "Usage: bash lite_release_package.sh input_path output_path"

input_path=$1
output_path=$2
version=`ls ${input_path}/android_aarch64/npu/mindspore-lite-*-*.tar.gz | awk -F'/' '{print $NF}' | cut -d"-" -f3`

android_release_package aarch32 npu
android_release_package aarch32 cpu
android_release_package aarch64 npu
android_release_package aarch64 gpu

ios_release_package
linux_release_package
windows_release_package
openharmony_release_package

echo "Create release package success!"
echo "=============================== end ==============================="
