#!/bin/bash
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

if [ $# != 6 ]
then
    echo "Usage: sh run_standalone_eval_ascend.sh [data_path][ckpt_url][ckpt_name][device_id][graph_conv_type][n_pred]"
exit 1
fi

export Data_path=$1
export Ckpt_path=$2
export Ckpt_name=$3
export Device_id=$4
export Graph_conv_type=$5
export N_pred=$6

python test.py --data_url=$Data_path   \
                --train_url=./checkpoint   \
                --run_distribute=False   \
                --run_modelarts=False   \
                --device_id=$Device_id  \
                --ckpt_url=$Ckpt_path   \
                --ckpt_name=$Ckpt_name  \
                --n_pred=$N_pred    \
                --graph_conv_type=$Graph_conv_type > test.log 2>&1 &
