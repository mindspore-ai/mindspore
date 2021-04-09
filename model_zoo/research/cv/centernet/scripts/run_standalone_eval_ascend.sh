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
echo "bash run_standalone_eval_ascend.sh DEVICE_ID RUN_MODE DATA_DIR LOAD_CHECKPOINT_PATH"
echo "for example of validation: bash run_standalone_eval_ascend.sh 0 val /path/coco_dataset /path/load_ckpt"
echo "for example of test: bash run_standalone_eval_ascend.sh 0 test /path/coco_dataset /path/load_ckpt"
echo "=============================================================================================================="
DEVICE_ID=$1
RUN_MODE=$2
DATA_DIR=$3
LOAD_CHECKPOINT_PATH=$4
mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

# install nms module from third party
if python -c "import nms" > /dev/null 2>&1
then
    echo "NMS module already exits, no need reinstall."
else
    echo "NMS module was not found, install it now..."
    git clone https://github.com/xingyizhou/CenterNet.git
    cd CenterNet/src/lib/external/ || exit
    make
    python setup.py install
    cd - || exit
    rm -rf CenterNet
fi

python ${PROJECT_DIR}/../eval.py  \
    --device_target=Ascend \
    --device_id=$DEVICE_ID \
    --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
    --data_dir=$DATA_DIR \
    --run_mode=$RUN_MODE \
    --visual_image=true \
    --enable_eval=true \
    --save_result_dir=./ > eval_log.txt 2>&1 &
