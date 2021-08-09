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

docker_image=$1
share_dir=$2
data_dir=$3
echo "$1"
echo "$2"
if [ -z "${docker_image}" ]; then
    echo "please input docker_image"
    exit 1
fi

if [ ! -d "${share_dir}" ]; then
    echo "please input share directory that contains dataset, models and codes"
    exit 1
fi


docker run -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --privileged \
    -v //usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v ${data_dir}:${data_dir}  \
    -v ${share_dir}:${share_dir} \
    ${docker_image} \
    /bin/bash
