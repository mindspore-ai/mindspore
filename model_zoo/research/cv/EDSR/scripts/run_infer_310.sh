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

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SCALE] [LOG_FILE] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, default: 0"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

model=$(get_real_path $1)
data_path=$(get_real_path $2)
scale=$3

if [[ $scale -ne "2" &&  $scale -ne "3" &&  $scale -ne "4" ]]; then
    echo "[SCALE] should be in [2,3,4]"
exit 1
fi

log_file="./run_infer.log"
if [ $# -gt 4 ]; then
    log_file=$4
fi
log_file=$(get_real_path $log_file)


device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

self_ensemble="True"

echo "***************** param *****************"
echo "mindir name: "$model
echo "dataset path: "$data_path
echo "scale: "$scale
echo "log file: "$log_file
echo "device id: "$device_id
echo "self_ensemble: "$self_ensemble
echo "***************** param *****************"

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

export PYTHONPATH=$PWD:$PYTHONPATH

function compile_app()
{
    echo "begin to compile app..."
    cd ./ascend310_infer || exit
    bash build.sh >> $log_file  2>&1
    cd -
    echo "finshi compile app"
}

function preprocess()
{
    echo "begin to preprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    python preprocess.py --data_path=$data_path --config_path=DIV2K_config.yaml --device_target=CPU --scale=$scale --self_ensemble=$self_ensemble >> $log_file 2>&1
    echo "finshi preprocess"
}

function infer()
{
    echo "begin to infer..."
    if [ $self_ensemble == "True" ]; then
        read_data_path=$data_path"/DIV2K_valid_LR_bicubic_AUG_self_ensemble/X"$scale
    else
        read_data_path=$data_path"/DIV2K_valid_LR_bicubic_AUG/X"$scale
    fi
    save_data_path=$data_path"/DIV2K_valid_SR_bin/X"$scale
    if [ -d $save_data_path ]; then
        rm -rf $save_data_path
    fi
    mkdir -p $save_data_path
    ./ascend310_infer/out/main --mindir_path=$model --dataset_path=$read_data_path --device_id=$device_id --save_dir=$save_data_path >> $log_file 2>&1
    echo "finshi infer"
}

function postprocess()
{
    echo "begin to postprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    python postprocess.py --data_path=$data_path --config_path=DIV2K_config.yaml --device_target=CPU --scale=$scale --self_ensemble=$self_ensemble >> $log_file 2>&1
    echo "finshi postprocess"
}

echo "" > $log_file
echo "read the log command: "
echo "    tail -f $log_file"

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed, check $log_file"
    exit 1
fi

preprocess
if [ $? -ne 0 ]; then
    echo "preprocess code failed, check $log_file"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed, check $log_file"
    exit 1
fi

postprocess
if [ $? -ne 0 ]; then
    echo "postprocess failed, check $log_file"
    exit 1
fi

cat $log_file | tail -n 3 | head -n 1
