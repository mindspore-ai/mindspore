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
    dst_pkg_name="mindspore-lite-${version}-android-${arch}"
    rm -rf ${dst_pkg_name}
    mv ${input_path}/android_${arch}/${device}/${dst_pkg_name}.tar.gz ${output_path}/release/android/${device}/${dst_pkg_name}.tar.gz
    cd ${output_path}/release/android/${device}/
    sha256sum ${dst_pkg_name}.tar.gz > ${dst_pkg_name}.tar.gz.sha256
}

function linux_release_package()
{

    dst_pkg_name="mindspore-lite-${version}-linux-x64"
    rm -rf ${dst_pkg_name}
    mkdir -p ${output_path}/release/linux/
    mv ${input_path}/ubuntu_x86/${dst_pkg_name}.tar.gz ${output_path}/release/linux/
    cd ${output_path}/release/linux/
    sha256sum ${dst_pkg_name}.tar.gz > ${dst_pkg_name}.tar.gz.sha256
}

function windows_release_package()
{
    pkg_name="mindspore-lite-${version}-win-x64"

    rm -rf ${pkg_name}
    mv  ${input_path}/windows_x64/avx/${pkg_name}.zip ${output_path}/release/windows/${dst_pkg_name}.zip
    cd ${output_path}/release/windows/
    sha256sum ${dst_pkg_name}.zip > ${dst_pkg_name}.zip.sha256
}

echo "============================== begin =============================="
echo "Usage: bash lite_release_package.sh input_path output_path"

input_path=$1
output_path=$2
version=`ls ${input_path}/android_aarch64/mindspore-lite-*-*.tar.gz | awk -F'/' '{print $NF}' | cut -d"-" -f3`

android_release_package aarch32
android_release_package aarch64
android_release_package aarch64 gpu
linux_release_package
windows_release_package

echo "Create release package success!"
echo "=============================== end ==============================="
