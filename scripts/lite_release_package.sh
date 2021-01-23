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

set -e

function verify_every_file() {
    for full_file in "$1"/*
    do
        if [ -d ${full_file} ]
        then
            verify_every_file ${full_file} $2
        else
            echo "check: ${full_file}"
            exist_and_equal="false"
            src_sha256=`sha256sum ${full_file} | cut -d" " -f1`
            file_name=$(basename ${full_file})
            find_result=`find $2 -name ${file_name} -type f`
            for same_name_file in ${find_result}
            do
                dst_sha256=`sha256sum ${same_name_file} | cut -d" " -f1`
                if [ ${src_sha256} == ${dst_sha256} ]
                then
                    echo "  dst: ${same_name_file}"
                    exist_and_equal="true"
                fi
            done
            if [ ${exist_and_equal} == "false" ]
            then
                echo "    check failed!"
                exit 1
            fi
        fi
    done
}

function android_release_package()
{
    for name in "train" "inference"
    do
        src_arm64_pkg_name="mindspore-lite-${version}-${name}-android-aarch64"
        src_arm32_pkg_name="mindspore-lite-${version}-${name}-android-aarch32"
        dst_android_pkg_name="mindspore-lite-${version}-${name}-android"

        tar -xzf ${input_path}/android_aarch64/${src_arm64_pkg_name}.tar.gz
        tar -xzf ${input_path}/android_aarch32/${src_arm32_pkg_name}.tar.gz

        # ARM32 and ARM64 have the same header file.
        mkdir -p ${dst_android_pkg_name}/minddata/
        cp -r ${src_arm64_pkg_name}/include/ ${dst_android_pkg_name}/
        cp -r ${src_arm64_pkg_name}/minddata/include/ ${dst_android_pkg_name}/minddata/
        cp ${src_arm64_pkg_name}/.commit_id ${dst_android_pkg_name}/

        # Executable files and dynamic libraries are different in different architectures.
        mkdir -p ${dst_android_pkg_name}/benchmark/aarch64/
        mkdir -p ${dst_android_pkg_name}/benchmark/aarch32/
        mkdir -p ${dst_android_pkg_name}/lib/aarch64/   
        mkdir -p ${dst_android_pkg_name}/lib/aarch32/
        mkdir -p ${dst_android_pkg_name}/minddata/lib/aarch64/
        mkdir -p ${dst_android_pkg_name}/minddata/lib/aarch32/
        cp ${src_arm64_pkg_name}/benchmark/* ${dst_android_pkg_name}/benchmark/aarch64/
        cp ${src_arm32_pkg_name}/benchmark/* ${dst_android_pkg_name}/benchmark/aarch32/
        cp ${src_arm64_pkg_name}/lib/* ${dst_android_pkg_name}/lib/aarch64/
        cp ${src_arm32_pkg_name}/lib/* ${dst_android_pkg_name}/lib/aarch32/
        cp ${src_arm64_pkg_name}/minddata/lib/* ${dst_android_pkg_name}/minddata/lib/aarch64/
        cp ${src_arm32_pkg_name}/minddata/lib/* ${dst_android_pkg_name}/minddata/lib/aarch32/

        if [ ${name} == "train" ]
        then
            mkdir -p ${dst_android_pkg_name}/benchmark_train/aarch64/
            mkdir -p ${dst_android_pkg_name}/benchmark_train/aarch32/
            mkdir -p ${dst_android_pkg_name}/minddata/third_party/libjpeg-turbo/lib/aarch64/
            mkdir -p ${dst_android_pkg_name}/minddata/third_party/libjpeg-turbo/lib/aarch32/
            cp ${src_arm64_pkg_name}/benchmark_train/* ${dst_android_pkg_name}/benchmark_train/aarch64/
            cp ${src_arm32_pkg_name}/benchmark_train/* ${dst_android_pkg_name}/benchmark_train/aarch32/
            cp ${src_arm64_pkg_name}/minddata/third_party/libjpeg-turbo/lib/* ${dst_android_pkg_name}/minddata/third_party/libjpeg-turbo/lib/aarch64/
            cp ${src_arm32_pkg_name}/minddata/third_party/libjpeg-turbo/lib/* ${dst_android_pkg_name}/minddata/third_party/libjpeg-turbo/lib/aarch32/
        fi
        mkdir -p ${dst_android_pkg_name}/third_party/hiai_ddk/lib/aarch64/
        cp -r ${src_arm64_pkg_name}/third_party/hiai_ddk/lib/* ${dst_android_pkg_name}/third_party/hiai_ddk/lib/aarch64/
        if [ ${name} == "inference" ]
        then
            # Copy java runtime to Android package
            cp ${input_path}/aar/* ${dst_android_pkg_name}
        fi

        mkdir -p ${output_path}/release/android/
        tar -czf ${output_path}/release/android/${dst_android_pkg_name}.tar.gz ${dst_android_pkg_name}
        cd ${output_path}/release/android/
        sha256sum ${dst_android_pkg_name}.tar.gz > ${dst_android_pkg_name}.tar.gz.sha256
        cd -

        verify_every_file ${src_arm64_pkg_name} ${dst_android_pkg_name}
        verify_every_file ${src_arm32_pkg_name} ${dst_android_pkg_name}
        rm -rf ${src_arm64_pkg_name}
        rm -rf ${src_arm32_pkg_name}
        rm -rf ${dst_android_pkg_name}
    done
}

function linux_release_package()
{
    mkdir -p ${output_path}/release/linux/
    cp ${input_path}/ubuntu_x86/mindspore-lite-${version}-converter-* ${output_path}/release/linux/
    cp ${input_path}/ubuntu_x86/mindspore-lite-${version}-inference-linux-x64-avx.tar.gz  ${output_path}/release/linux/mindspore-lite-${version}-inference-linux-x64.tar.gz
    cp ${input_path}/ubuntu_x86/mindspore-lite-${version}-inference-linux-x64-avx.tar.gz.sha256  ${output_path}/release/linux/mindspore-lite-${version}-inference-linux-x64.tar.gz.sha256
    cp ${input_path}/ubuntu_x86/mindspore-lite-${version}-train-* ${output_path}/release/linux/
}

function windows_release_package()
{
    mkdir -p ${output_path}/release/windows/
    cp ${input_path}/windows_x64/mindspore-lite-${version}-converter-* ${output_path}/release/windows/
    cp ${input_path}/windows_x64/mindspore-lite-${version}-inference-win-x64-avx.zip ${output_path}/release/windows/mindspore-lite-${version}-inference-win-x64.zip
    cp ${input_path}/windows_x64/mindspore-lite-${version}-inference-win-x64-avx.zip.sha256 ${output_path}/release/windows/mindspore-lite-${version}-inference-win-x64.zip.sha256
}

echo "============================== begin =============================="
echo "Usage: bash lite_release_package.sh input_path output_path"

input_path=$1
output_path=$2
version=`ls ${input_path}/android_aarch64/mindspore-lite-*-inference-*.tar.gz | awk -F'/' '{print $NF}' | cut -d"-" -f3`

android_release_package
linux_release_package
windows_release_package

echo "Create release package success!"
echo "=============================== end ==============================="
