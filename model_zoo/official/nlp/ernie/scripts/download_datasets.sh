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
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh download_dataset.sh [DATA_TYPE]"
    echo "for example: sh scripts/download_dataset.sh.sh pretrain"
    echo "TASK_TYPE including [pretrain, finetune]"
    echo "=============================================================================================================="
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CUR_DIR=`pwd`
DATA_PATH=${CUR_DIR}/data

TASK_TYPE=$1
case $TASK_TYPE in
  "pretrain")
    DATA_URL=https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
    wget --no-check-certificate ${DATA_URL} 
    python -m wikiextractor.WikiExtractor -o extracted zhwiki-latest-pages-articles.xml.bz2
    /bin/rm zhwiki-latest-pages-articles.xml.bz2
    if [ ! -d $DATA_PATH ]
    then
        mkdir ${CUR_DIR}/data
        mv ${CUR_DIR}/extracted ${CUR_DIR}/data/extracted
    else
        mv ${CUR_DIR}/extracted ${CUR_DIR}/data/extracted
    exit 1
    fi
    ;;
  "finetune")
    DATA_URL=https://ernie.bj.bcebos.com/task_data_zh.tgz
    wget --no-check-certificate ${DATA_URL}
    tar xvf task_data_zh.tgz
    /bin/rm task_data_zh.tgz
    if [ ! -d $DATA_PATH ]
    then
        mv ${CUR_DIR}/task_data ${CUR_DIR}/data
    else
        mv ${CUR_DIR}/task_data/* ${CUR_DIR}/data/
        rm -rf ${CUR_DIR}/task_data/*
    exit 1
    fi    
    ;;
  esac


