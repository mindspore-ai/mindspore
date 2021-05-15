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
    echo "bash scripts/run_standalone_eval.sh [DATA_PATH] [CHECKPOINT]"
    echo "for example: bash run_standalone_eval.sh /path/to/data/ /path/to/checkpoint/"
    echo "=============================================================================================================="
fi

if [ $# != 2 ]
then
    echo "Usage: sh run_eval_ascend.sh [DATA_PATH] [CHECKPOINT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)
CHECKPOINT_FILE_PATH=$(get_real_path $2)
echo $PATH1
echo $CHECKPOINT_FILE_PATH

if [ ! -d $PATH1 ]
then
    echo "error: PATH1=$PATH1 is not a path"
exit 1
fi

if [ ! -f $CHECKPOINT_FILE_PATH ]
then
    echo "error: CHECKPOINT_FILE_PATH=$CHECKPOINT_FILE_PATH is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=0
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cd ./eval || exit
echo "start eval for checkpoint file: ${CHECKPOINT_FILE_PATH}"
python eval.py --data_path=$PATH1 --checkpoint_file_path=$CHECKPOINT_FILE_PATH > eval.log 2>&1 &
echo "end eval for checkpoint file: ${CHECKPOINT_FILE_PATH}"
cd ..
