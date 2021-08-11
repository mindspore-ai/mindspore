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
echo "sh run_distributed_train.sh DATASET_PATH RANK_TABLE_PATH"
echo "for example: sh run_distributed_train.sh /home/workspace/ag /home/workspace/rank_table_file.json ag"
echo "It is better to use absolute path."
echo "Please pay attention that the dataset should corresponds to dataset_name"
echo "=============================================================================================================="
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $3 != "ag" ] && [ $3 != "dbpedia" ] && [ $3 != "yelp_p" ]
then
  echo "Unrecognized dataset name, the name can choose from [ag, dbpedia, yelp_p]"
exit 1
fi

DATASET=$(get_real_path $1)
echo $DATASET
RANK_TABLE_PATH=$(get_real_path $2)
if [ ! -d $DATASET ]
then
  echo "Error: DATA_PATH=$DATASET is not a file"
exit 1
fi
current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_PATH


echo $RANK_TABLE_FILE
export RANK_SIZE=8
export DEVICE_NUM=8

if [ $# -ge 1 ]; then
  if [ $3 == 'ag' ]; then
    DATANAME='ag'
  elif [ $3 == 'dbpedia' ]; then
    DATANAME='dbpedia'
  elif [ $3 == 'yelp_p' ]; then
    DATANAME='yelp_p'
  else
    echo "Unrecognized dataset name,he name can choose from [ag, dbpedia, yelp_p]"
    exit 1
  fi
fi

config_path="./${DATANAME}_config.yaml"
echo "config path is : ${config_path}"

for((i=0;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py ./
    cp ../../*.yaml ./
    cp -r ../../src ./
    cp -r ../../model_utils ./
    cp -r ../*.sh ./
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    python ../../train.py --config_path $config_path --dataset_path $DATASET --data_name $DATANAME > log_fasttext.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
