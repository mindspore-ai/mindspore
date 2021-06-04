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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh DATA_PATH"
echo "for example: bash run_distribute_train_gpu.sh /path/ImageNet2012/train"
echo "=============================================================================================================="

DATA_PATH=$1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
config_path=$(get_real_path "./imagenet2012_config.yaml")

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout \
  python train.py  \
    --config_path=$config_path \
    --device_target="GPU" \
    --dataset="imagenet2012" \
    --is_distributed=1 \
    --data_dir=$DATA_PATH  > output.train.log 2>&1 &
