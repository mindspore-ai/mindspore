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

ulimit -u unlimited
export DATA_PATH=$1
export DATA_TEST=$2
export MODEL=$3
export TASK_ID=$4

if [[ $TASK_ID -lt 3 ]]; then
    mkdir ./run_eval$TASK_ID
    cp -r ../src ./run_eval$TASK_ID
    cp ../*.py ./run_eval$TASK_ID
    echo "start evaluation for Task $TASK_ID, device $DEVICE_ID"
    cd ./run_eval$TASK_ID ||exit
    env > env$TASK_ID.log
    SCALE=$[$TASK_ID+2]
    python eval.py --dir_data $DATA_PATH --data_test $DATA_TEST --test_only --ext img --pth_path $MODEL --task_id $TASK_ID --scale $SCALE > log 2>&1 &
fi

if [[ $TASK_ID -eq 3 ]]; then
    mkdir ./run_eval$TASK_ID
    cp -r ../src ./run_eval$TASK_ID
    cp ../*.py ./run_eval$TASK_ID
    echo "start evaluation for Task $TASK_ID, device $DEVICE_ID"
    cd ./run_eval$TASK_ID ||exit
    env > env$TASK_ID.log
    python eval.py --dir_data $DATA_PATH --data_test $DATA_TEST --test_only --ext img --pth_path $MODEL --task_id $TASK_ID --scale 1  --derain > log 2>&1 &
fi

if [[ $TASK_ID -eq 4 ]]; then
    mkdir ./run_eval$TASK_ID
    cp -r ../src ./run_eval$TASK_ID
    cp ../*.py ./run_eval$TASK_ID
    echo "start evaluation for Task $TASK_ID, device $DEVICE_ID"
    cd ./run_eval$TASK_ID ||exit
    env > env$TASK_ID.log
    python eval.py --dir_data $DATA_PATH --data_test $DATA_TEST --test_only --ext img --pth_path $MODEL --task_id $TASK_ID --scale 1  --denoise --sigma 30  > log 2>&1 &
fi

if [[ $TASK_ID -eq 5 ]]; then
    mkdir ./run_eval$TASK_ID
    cp -r ../src ./run_eval$TASK_ID
    cp ../*.py ./run_eval$TASK_ID
    echo "start evaluation for Task $TASK_ID, device $DEVICE_ID"
    cd ./run_eval$TASK_ID ||exit
    env > env$TASK_ID.log
    python eval.py --dir_data $DATA_PATH --data_test $DATA_TEST --test_only --ext img --pth_path $MODEL --task_id $TASK_ID --scale 1  --denoise --sigma 50  > log 2>&1 &
fi