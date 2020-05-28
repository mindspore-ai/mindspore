# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_standalone_pretrain.sh DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash run_standalone_train.sh 0 40 /path/zh-wiki/ "
echo "=============================================================================================================="
 
DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3
 
mkdir -p ms_log 
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python train.py  \
    --distribute="false" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --enable_save_ckpt="true" \
    --checkpoint_url="/store1/deeplabv3/deeplabv3_split_url/train/checkpoint/CKP-12_732.ckpt" \
    --save_checkpoint_steps=10000 \
    --save_checkpoint_num=1 \
    --data_url=$DATA_DIR > log.txt 2>&1 &