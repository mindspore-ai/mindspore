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
    arch=$1
    device=$2
    src_inference_pkg_name="mindspore-lite-${version}-inference-android-${arch}"
    src_train_pkg_name="mindspore-lite-${version}-train-android-${arch}"
    dst_pkg_name="mindspore-lite-${version}-android-${arch}"

    rm -rf ${src_inference_pkg_name}
    rm -rf ${src_train_pkg_name}
    rm -rf ${dst_pkg_name}
    tar -xzf ${input_path}/android_${arch}/${device}/${src_inference_pkg_name}.tar.gz
    tar -xzf ${input_path}/android_${arch}/${device}/${src_train_pkg_name}.tar.gz

    cp -r ${src_train_pkg_name}/tools/benchmark_train/ ${src_inference_pkg_name}/tools/
    cp -r ${src_train_pkg_name}/train/ ${src_inference_pkg_name}/
    mkdir -p ${output_path}/release/android/${device}/
    mv ${src_inference_pkg_name} ${dst_pkg_name}
    # Copy java runtime to Android package
    cp ${input_path}/aar/avx/mindspore-lite-*maven*.zip ${dst_pkg_name}
    tar -czf ${output_path}/release/android/${device}/${dst_pkg_name}.tar.gz ${dst_pkg_name}
    cd ${output_path}/release/android/${device}/
    sha256sum ${dst_pkg_name}.tar.gz > ${dst_pkg_name}.tar.gz.sha256
    cd -

    verify_every_file ${src_train_pkg_name}/tools/benchmark_train/ ${dst_pkg_name}
    verify_every_file ${src_train_pkg_name}/train/ ${dst_pkg_name}

    rm -rf ${src_train_pkg_name}
    rm -rf ${dst_pkg_name}
}

function linux_release_package()
{
    src_inference_pkg_name="mindspore-lite-${version}-inference-linux-x64"
    src_train_pkg_name="mindspore-lite-${version}-train-linux-x64"
    src_jar_pkg_name="mindspore-lite-${version}-inference-linux-x64-jar"
    dst_pkg_name="mindspore-lite-${version}-linux-x64"

    rm -rf ${src_inference_pkg_name}
    rm -rf ${src_train_pkg_name}
    rm -rf ${src_jar_pkg_name}
    rm -rf ${dst_pkg_name}
    tar -xzf ${input_path}/ubuntu_x86/avx/${src_inference_pkg_name}.tar.gz
    tar -xzf ${input_path}/ubuntu_x86/${src_train_pkg_name}.tar.gz
    tar -xzf ${input_path}/aar/avx/${src_jar_pkg_name}.tar.gz

    cp -r ${src_train_pkg_name}/tools/benchmark_train/ ${src_inference_pkg_name}/tools/
    cp -r ${src_train_pkg_name}/train/ ${src_inference_pkg_name}/
    cp -r ${src_jar_pkg_name}/jar/ ${src_inference_pkg_name}/inference/lib/

    mkdir -p ${output_path}/release/linux/
    mv ${src_inference_pkg_name} ${dst_pkg_name}
    tar -czf ${output_path}/release/linux/${dst_pkg_name}.tar.gz ${dst_pkg_name}
    cd ${output_path}/release/linux/
    sha256sum ${dst_pkg_name}.tar.gz > ${dst_pkg_name}.tar.gz.sha256
    cd -

    verify_every_file ${src_train_pkg_name}/tools/benchmark_train/ ${dst_pkg_name}
    verify_every_file ${src_train_pkg_name}/train/ ${dst_pkg_name}
    verify_every_file ${src_jar_pkg_name}/ ${dst_pkg_name}
    rm -rf ${src_train_pkg_name}
    rm -rf ${src_jar_pkg_name}
    rm -rf ${dst_pkg_name}
}

function windows_release_package()
{
    src_inference_pkg_name="mindspore-lite-${version}-inference-win-x64"
    dst_pkg_name="mindspore-lite-${version}-win-x64"

    rm -rf ${src_inference_pkg_name}
    rm -rf ${dst_pkg_name}
    unzip ${input_path}/windows_x64/avx/${src_inference_pkg_name}.zip

    mv ${src_inference_pkg_name} ${dst_pkg_name}
    mkdir -p ${output_path}/release/windows/
    zip -r ${output_path}/release/windows/${dst_pkg_name}.zip ${dst_pkg_name}
    cd ${output_path}/release/windows/
    sha256sum ${dst_pkg_name}.zip > ${dst_pkg_name}.zip.sha256
    cd -
    rm -rf ${dst_pkg_name}
}

echo "============================== begin =============================="
echo "Usage: bash lite_release_package.sh input_path output_path"

input_path=$1
output_path=$2
version=`ls ${input_path}/android_aarch64/mindspore-lite-*-inference-*.tar.gz | awk -F'/' '{print $NF}' | cut -d"-" -f3`

android_release_package aarch32
android_release_package aarch64
android_release_package aarch64 gpu
linux_release_package
windows_release_package

echo "Create release package success!"
echo "=============================== end ==============================="
