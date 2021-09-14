#!/usr/bin/env bash

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

docker_image=$1
data_path=$2

function show_help() {
    echo "Usage: docker_start.sh docker_image data_path"
}

function param_check() {
    if [ -z "${docker_image}" ]; then
        echo "please input docker_image"
        show_help
        exit 1
    fi

    if [ -z "${data_path}" ]; then
        echo "please input data_path"
        show_help
        exit 1
    fi
}

param_check

docker run -it \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${data_path}:${data_path} \
  ${docker_image} \
  /bin/bash
