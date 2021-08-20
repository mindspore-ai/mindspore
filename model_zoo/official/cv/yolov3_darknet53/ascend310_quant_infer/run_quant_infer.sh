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

if [ $# -lt 5 ]; then
    echo "Usage: bash run_quant_infer.sh [AIR_PATH] [DATA_PATH] [IMAGE_ID] [IMAGE_SHAPE] [ANN_FILE]"
    echo "Example: bash run_quant_infer.sh ./yolov3_quant.air ./image_bin ./image_id.npy ./image_shape.npy \
./instances_val2014.json"
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
id_path=$(get_real_path $3)
shape_path=$(get_real_path $4)
ann_path=$(get_real_path $5)

echo "air name: "$model
echo "dataset path: "$data_path
echo "id path: "$id_path
echo "shape path: "$shape_path
echo "ann path: "$ann_path

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export PATH=$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/fwkacllib/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/atc/lib64:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function air_to_om()
{
    atc --input_format=NCHW --framework=1 --model=$model --output=yolov3_quant --soc_version=Ascend310 &> atc.log
}

function compile_app()
{
    bash ./src/build.sh &> build.log
}

function infer()
{
    if [ -d result ]; then
        rm -rf ./result
    fi
    mkdir result
    ./out/main ./yolov3_quant.om $data_path &> infer.log
}

function cal_acc()
{
    python3.7 ./acc.py --result_path=./result --annFile=$ann_path --image_shape=$shape_path \
    --image_id=$id_path &> acc.log
}

echo "start atc================================================"
air_to_om
if [ $? -ne 0 ]; then
    echo "air to om code failed"
    exit 1
fi

echo "start compile============================================"
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

echo "start infer=============================================="
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi

echo "start calculate acc======================================"
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi