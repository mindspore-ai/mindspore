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

    [ -n "${pkg_name}" ] && rm -rf ${pkg_name}
    tar -xzf ${input_path}/android_${arch}/${device}/${pkg_name}.tar.gz
    # Copy java runtime to Android package
    cp ${input_path}/aar/mindspore-lite-*.aar ${pkg_name}

    mkdir -p ${output_path}/release/android/${device}/
    tar -czf ${output_path}/release/android/${device}/${pkg_name}.tar.gz ${pkg_name}
    [ -n "${pkg_name}" ] && rm -rf ${pkg_name}
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
    mkdir -p ${output_path}/release/linux/x86_64/
    mkdir -p ${output_path}/release/linux/x86_64/tensorrt/
    mkdir -p ${output_path}/release/linux/aarch64/
    mkdir -p ${output_path}/release/linux/x86_64/ascend/
    mkdir -p ${output_path}/release/linux/aarch64/ascend/
    mkdir -p ${output_path}/release/linux/x86_64/server/
    mkdir -p ${output_path}/release/linux/aarch64/server/
    mkdir -p ${output_path}/release/linux/x86_64/cloud_fusion/
    mkdir -p ${output_path}/release/linux/aarch64/cloud_fusion/
    mkdir -p ${output_path}/release/none/cortex_m7

    cp ${input_path}/none_cortex-m/mindspore*cortex-m7.tar.gz* ${output_path}/release/none/cortex_m7/
    cp ${input_path}/centos_x86/avx/mindspore*.whl* ${output_path}/release/linux/x86_64/
    cp ${input_path}/centos_x86/avx/mindspore*.tar.gz* ${output_path}/release/linux/x86_64/
    cp ${input_path}/linux_aarch64/mindspore*.whl* ${output_path}/release/linux/aarch64/
    cp ${input_path}/linux_aarch64/mindspore*.tar.gz* ${output_path}/release/linux/aarch64/
    cp ${input_path}/centos_x86/ascend/mindspore*.whl* ${output_path}/release/linux/x86_64/ascend/
    cp ${input_path}/centos_x86/ascend/mindspore*.tar.gz* ${output_path}/release/linux/x86_64/ascend/
    # the lite Ascend package has been replaced with a refactored version, which will be added later
    # cp ${input_path}/linux_aarch64/ascend/mindspore*.whl* ${output_path}/release/linux/aarch64/ascend/
    # cp ${input_path}/linux_aarch64/ascend/mindspore*.tar.gz* ${output_path}/release/linux/aarch64/ascend/
    cp ${input_path}/centos_x86/tensorrt/mindspore*.whl* ${output_path}/release/linux/x86_64/tensorrt/
    cp ${input_path}/centos_x86/tensorrt/mindspore*.tar.gz* ${output_path}/release/linux/x86_64/tensorrt/
    cp -r ${input_path}/centos_x86/server/* ${output_path}/release/linux/x86_64/server/
    cp -r ${input_path}/linux_aarch64/server/* ${output_path}/release/linux/aarch64/server/
    cp -r ${input_path}/centos_x86/cloud_fusion/* ${output_path}/release/linux/x86_64/cloud_fusion/
    cp -r ${input_path}/linux_aarch64/cloud_fusion/* ${output_path}/release/linux/aarch64/cloud_fusion/

    cp -r ${input_path}/linux_aarch32/nnie/Hi* ${output_path}/release/linux/nnie/
    cp -r ${input_path}/linux_aarch64/nnie/Hi* ${output_path}/release/linux/nnie/
    cp ${input_path}/centos_x86/nnie/Hi3516D/*.tar.gz* ${output_path}/release/linux/nnie/
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
version=$(ls ${input_path}/android_aarch64/npu/mindspore-lite-*-*.tar.gz | awk -F'/' '{print $NF}' | cut -d"-" -f3)

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
