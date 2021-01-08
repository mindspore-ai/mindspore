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
echo "sh create_dataset.sh SOURCE_DATASET_PATH DATASET_NAME"
echo "for example: sh create_dataset.sh /home/workspace/ag_news_csv ag"
echo "DATASET_NAME including ag, dbpedia, and yelp_p"
echo "It is better to use absolute path."
echo "=============================================================================================================="
ulimit -u unlimited
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
SOURCE_DATASET_PATH=$(get_real_path $1)
DATASET_NAME=$2

export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ $DATASET_NAME == 'ag' ];
then
  echo "Begin to process ag news data"
  if [ -d "ag" ];
  then
    rm -rf ./ag
  fi
  mkdir ./ag
  cd ./ag || exit
  echo "start data preprocess for device $DEVICE_ID"
  python ../../src/dataset.py  --train_file $SOURCE_DATASET_PATH/train.csv --test_file $SOURCE_DATASET_PATH/test.csv --class_num 4 --max_len 467 --bucket [64,128,467] --test_bucket [467]
  cd ..
fi

if [ $DATASET_NAME == 'dbpedia' ];
then
  echo "Begin to process dbpedia data"
  if [ -d "dbpedia" ];
  then
    rm -rf ./dbpedia
  fi
  mkdir ./dbpedia
  cd ./dbpedia || exit
  echo "start data preprocess for device $DEVICE_ID"
  python ../../src/dataset.py  --train_file $SOURCE_DATASET_PATH/train.csv --test_file $SOURCE_DATASET_PATH/test.csv --class_num 14 --max_len 3013 --bucket [64,128,256,512,3013] --test_bucket [64,128,256,512,1120]
  cd ..
fi

if [ $DATASET_NAME == 'yelp_p' ];
then
  echo "Begin to process ag news data"
  if [ -d "yelp_p" ];
  then
    rm -rf ./yelp_p
  fi
  mkdir ./yelp_p
  cd ./yelp_p || exit
  echo "start data preprocess for device $DEVICE_ID"
  python ../../src/dataset.py  --train_file $SOURCE_DATASET_PATH/train.csv --test_file $SOURCE_DATASET_PATH/test.csv --class_num 2 --max_len 2955 --bucket [64,128,256,512,2955] --test_bucket [64,128,256,512,2955]
  cd ..
fi




