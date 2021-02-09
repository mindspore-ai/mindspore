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

stage=0
days=7
iter=5
device_id=0
n_worker=128
work_dir="/opt/npu/me_monitor"
me_report_path=$work_dir/logs/ME_report_daily.xlsx
log_dir=logs_$(date "+%m%d-%H%M")
log_path=$work_dir/logs/$log_dir
ms_master="https://gitee.com/mindspore/mindspore.git"
log_data="data.json"
ci_mode=true

set -e
set -o pipefail

# parse arguments from command line
while getopts "s:d:i:l:" args
do
    case $args in
        s)
            stage=$OPTARG
            ;;
        d)
            days=$OPTARG
            ;;
        i)
            iter=$OPTARG
            ;;
        l)
            log_dir=$OPTARG
            log_path=$work_dir/logs/$log_dir
            ;;
        ?)
            echo "unknown argument"
            exit 1
            ;;
    esac
done

source env.sh
export DEVICE_ID=$device_id
echo "Args: days=$days, iter=$iter, log_path=$log_path"
cd $work_dir

echo $WORKSPACE
WORKSPACE=/home/jenkins-slave/workspace/MindSpore_Network_reid_compile_performance
echo $WORKSPACE

if [ $stage -le 1 ]; then
    echo ""
    echo "===========Stage 1: Fetching latest mindspore from master==========="
    if [ -d mindspore ]; then
        rm -rf mindspore
    fi
    git clone $ms_master
fi

if [ $stage -le 2 ]; then
    echo ""
    echo "===========Stage 2: Building mindspore==========="
    cd $work_dir/mindspore
    bash build.sh -e ascend -j $n_worker -p on
fi

if [ $stage -le 3 ]; then
    echo ""
    echo "===========Stage 3: Compiling networks==========="
    cd $work_dir
    mkdir -p $log_path

    # Compiling ReID-8
    # split resource-consuming task from others
    for count in $(seq 1 $iter); do
        echo "[INFO] Compiling ReID-8p, iteration $count"
        if [ -d reid$count ]; then
            rm -rf reid$count
        fi
        mkdir reid$count
        cd reid$count
        bash $work_dir/faceReidToMe/dist_env/env_26/dist_env_26.sh
        for num in {0..7}; do
            cp device_$num/test_reid_stage123_1024node_graphdata_dynamiclossscale_log$num.log $log_path/reid_${count}_${num}.log
        done
        cd $work_dir
        mv reid$count $log_path
    done

    # Compiling BERT
    cd $work_dir
    for count in $(seq 1 $iter); do
        echo "[INFO] Compiling BERT, iteration $count"
        pytest -s mindspore/tests/perf_test/bert/test_bert_train.py::test_bert_train | tee $log_path/bert$count.log
    done
    
    # Compiling ResNet50
    for count in $(seq 1 $iter); do
        echo "[INFO] Compiling ResNet50, iteration $count"
        pytest -s mindspore/tests/perf_test/test_resnet_train.py::test_train_step | tee $log_path/resnet$count.log
    done

    # Compiling GPT
    for count in $(seq 1 $iter); do
        echo "[INFO] Compiling GPT, iteration $count"
        cd gpt
        bash scripts/run_standalone_train.sh 0 1 $work_dir/gpt_data | tee $log_path/gpt$count.log
    done
fi

if [ $stage -le 4 ]; then
    echo ""
    echo "===========Stage 4: Processing log files==========="
    cd $work_dir
    python process_data.py $me_report_path $log_path $iter $log_path/$log_data
fi

if [ $stage -le 5 ]; then
    echo ""
    echo "===========Stage 5: Generating reports==========="
    if [ ! -d $log_path/reports ]; then
        mkdir $log_path/reports
    fi
    python generate_report.py $log_path $log_path/$log_data $me_report_path $days

    if [ $ci_mode ]; then
        echo "copying file to artifacts"
        mkdir -p ${WORKSPACE}/archive 
        cp $log_path/reports/* ${WORKSPACE}/archive
    fi
fi
