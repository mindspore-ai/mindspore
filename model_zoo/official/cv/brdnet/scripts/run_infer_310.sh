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

if [ $# != 6 ]; then
    echo "Usage: sh run_infer_310.sh [model_path] [data_path]" \
       "[noise_image_path] [sigma] [channel] [device_id]"
exit 1
fi

get_real_path_name() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)/"
    fi
}

model=$(get_real_path_name $1)
data_path=$(get_real_path $2)
noise_image_path=$(get_real_path $3)
sigma=$4
channel=$5
device_id=$6

echo "model path: "$model
echo "dataset path: "$data_path
echo "noise image path: "$noise_image_path
echo "sigma: "$sigma
echo "channel: "$channel 
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export PATH=$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function compile_app()
{
    cd ../ascend310_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log
}

function preprocess()
{
    cd ../ || exit
    echo "\nstart preprocess"
    echo "waitting for preprocess finish..."
    python3.7 preprocess.py --out_dir=$noise_image_path --image_path=$data_path --channel=$channel --sigma=$sigma > preprocess.log 2>&1
    echo "preprocess finished! you can see the log in preprocess.log!"
}

function infer()
{
    cd ./scripts || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    echo "\nstart infer..."
    echo "waitting for infer finish..."
    ../ascend310_infer/out/main --model_path=$model --dataset_path=$noise_image_path --device_id=$device_id > infer.log 2>&1
    echo "infer finished! you can see the log in infer.log!"
}

function cal_psnr()
{
    echo "\nstart calculate PSNR..."
    echo "waitting for calculate finish..."
    python3.7 ../cal_psnr.py --image_path=$data_path --output_path=./result_Files --channel=$channel > psnr.log 2>&1
    echo "calculate finished! you can see the log in psnr.log\n"
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

preprocess
if [ $? -ne 0 ]; then
    echo "execute preprocess failed"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_psnr
if [ $? -ne 0 ]; then
    echo "calculate psnr failed"
    exit 1
fi
