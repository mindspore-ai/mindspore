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

prepare_third_party() {
  dpico_third_party=${mindspore_lite_top_dir}/tools/benchmark/dpico/third_party
  rm -rf ${dpico_third_party} || exit 1
  mkdir -p ${dpico_third_party} || exit 1
  cd ${mindspore_top_dir}/output || exit 1
  file_name=$(ls *tar.gz)
  tar_name=${file_name%%.tar.gz}
  tar xzvf ${tar_name}.tar.gz || exit 1
  cd ..
  cp -rf ${mindspore_top_dir}/output/${tar_name}/runtime/ ${dpico_third_party} || exit 1
}

# Build arm64 for dpico
make_dpico_benchmark_package() {
  cd ${mindspore_top_dir}/output || exit 1
  file_name=$(ls *tar.gz)
  tar_name=${file_name%%.tar.gz}
  dpico_sd3403_release_path=${mindspore_top_dir}/output/${tar_name}/providers/SD3403/
  mkdir -p ${dpico_sd3403_release_path}
  dpico_benchmark_path=${mindspore_top_dir}/mindspore/lite/build/tools/benchmark
  cp ${dpico_benchmark_path}/dpico/libdpico_acl_adapter.so ${dpico_sd3403_release_path} || exit 1
  echo "install dpico adapter so success."
  rm ${tar_name}.tar.gz || exit 1
  tar -zcf ${tar_name}.tar.gz ${tar_name} || exit 1
  rm -rf ${tar_name} || exit 1
  sha256sum ${tar_name}.tar.gz > ${tar_name}.tar.gz.sha256 || exit 1
  echo "generate dpico package success!"
  cd ${basepath}
  rm -rf ${dpico_third_party} || exit 1
}

basepath=$(pwd)
echo "basepath is ${basepath}"
#set -e
mindspore_top_dir=${basepath}
mindspore_lite_top_dir=${mindspore_top_dir}/mindspore/lite

while getopts "t:" opt; do
    case ${opt} in
        t)
            task=${OPTARG}
            echo "compile task is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

if [[ ${task} == "prepare_third_party" ]]; then
    prepare_third_party
    if [ $? -eq 1 ]; then
      echo "prepare third party failed"
      return 1
    fi
else
    echo "start make package for dpico..."
    make_dpico_benchmark_package &
    make_dpico_benchmark_package_pid=$!
    sleep 1

    wait ${make_dpico_benchmark_package_pid}
    make_dpico_benchmark_package_status=$?
    exit ${make_dpico_benchmark_package_status}
fi
