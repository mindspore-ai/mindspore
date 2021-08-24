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

if [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_distribute_train_s16_r1_gpu.sh /PATH/TO/MINDRECORD_NAME /PATH/TO/PRETRAIN_MODEL"
    echo "for example:"
    echo "bash run_distribute_train_s16_r1_gpu.sh \
      voc2012/mindrecord_train/vocaug_mindrecord0 resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt"
    echo "It is better to use absolute path."
    echo "=============================================================================================================="
exit 1
fi

DATA_FILE=$1
CKPT_PRE_TRAINED=$2

ulimit -c unlimited
export SLOG_PRINT_TO_STDOUT=0

export RANK_SIZE=8
export GLOG_v=2

train_path=s16_train
if [ -d ${train_path} ]; then
  rm -rf ${train_path}
fi
mkdir -p ${train_path}
mkdir ${train_path}/ckpt
cp ../*.py  ${train_path}
cp -r ../src  ${train_path}
cd ${train_path} || exit

mpirun -allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
python ./train.py --train_dir=${train_path}/ckpt  \
                  --data_file=$DATA_FILE  \
                  --train_epochs=300  \
                  --batch_size=16  \
                  --crop_size=513  \
                  --base_lr=0.04  \
                  --lr_type=cos  \
                  --min_scale=0.5  \
                  --max_scale=2.0  \
                  --ignore_label=255  \
                  --num_classes=21  \
                  --model=DeepLabV3plus_s16  \
                  --ckpt_pre_trained=$CKPT_PRE_TRAINED  \
                  --is_distributed  \
                  --save_steps=410  \
                  --keep_checkpoint_max=200 \
                  --device_target="GPU"  >log 2>&1 &
