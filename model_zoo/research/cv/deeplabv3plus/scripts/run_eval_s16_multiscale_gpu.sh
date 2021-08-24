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
if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval_s16_multiscale_gpu.sh /PATH/TO/DATA /PATH/TO/DATA_lst.txt /PATH/TO/PRETRAIN_MODEL DEVICE_ID"
    echo "for example:"
    echo "bash run_eval_s16_multiscale_gpu.sh \
      voc2012/VOCdevkit/VOC2012 voc2012/voc_val_lst.txt ckpt/DeepLabV3plus_s16-300_82.ckpt 0"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
    exit 1
fi

DATA_ROOT=$1
DATA_LST=$2
CKPT_PATH=$3
export CUDA_VISIBLE_DEVICES=$4

export SLOG_PRINT_TO_STDOUT=0
eval_path=s16_multiscale_eval
if [ -d ${eval_path} ]; then
  rm -rf ${eval_path}
fi
mkdir -p ${eval_path}
cp ../*.py  ${eval_path}
cp -r ../src  ${eval_path}
cd ${eval_path} || exit

python ./eval.py  --data_root=$DATA_ROOT  \
                  --data_lst=$DATA_LST  \
                  --batch_size=16  \
                  --crop_size=513  \
                  --ignore_label=255  \
                  --num_classes=21  \
                  --model=DeepLabV3plus_s16  \
                  --scales=0.5  \
                  --scales=0.75  \
                  --scales=1.0  \
                  --scales=1.25  \
                  --scales=1.75  \
                  --freeze_bn  \
                  --ckpt_path=$CKPT_PATH \
                  --device_target="GPU" \
                  --device_id=0 > eval_log 2>&1 &