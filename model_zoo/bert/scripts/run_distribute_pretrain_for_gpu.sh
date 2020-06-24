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
echo "Please run the scipt as: "
echo "bash run_distribute_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR"
echo "for example: bash run_distribute_pretrain.sh 8 40 /path/zh-wiki/ /path/Schema.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
EPOCH_SIZE=$2
DATA_DIR=$3
SCHEMA_DIR=$4

mpirun --allow-run-as-root -n $RANK_SIZE \
	python run_pretrain.py				\
		--device_target="GPU"			\
		--distribute="true"				\
		--epoch_size=$EPOCH_SIZE		\
		--enable_save_ckpt="true"		\
		--enable_lossscale="false"		\
		--do_shuffle="true"				\
		--enable_data_sink="true"		\
		--data_sink_steps=1				\
		--load_checkpoint_path=""			\
		--save_checkpoint_steps=10000	\
		--save_checkpoint_num=1			\
		--data_dir=$DATA_DIR			\
		--schema_dir=$SCHEMA_DIR > log.txt 2>&1 &

