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

if [ $# -lt 1 ]; then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_local_train.sh RANK_TABLE_FILE [OTHER_ARGS]"
    echo "OTHER_ARGS will be passed to the training scripts directly,"
    echo "for example: bash run_local_train.sh /path/hccl.json --data_dir=/path/data_dir --epochs=40"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
    exit 1
fi

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
OTHER_ARGS=$*
echo ${OTHER_ARGS[0]}

python3 ${BASE_PATH}/ascend_distributed_launcher/get_distribute_pretrain_cmd.py \
    --run_script_path=${BASE_PATH}/../train.py \
    --hccl_config_dir=$1 \
    --hccl_time_out=600 \
    --args="$*" \
    --cmd_file=distributed_cmd.sh

bash distributed_cmd.sh
