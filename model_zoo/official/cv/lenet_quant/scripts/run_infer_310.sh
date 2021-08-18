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

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [AIR_PATH] [DATA_PATH] [LABEL_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
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
label_path=$(get_real_path $3)

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "air name: "$model
echo "dataset path: "$data_path
echo "label path: "$label_path
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export PATH=$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/fwkacllib/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$PYTHONPATH:$TBE_IMPL_PATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function air_to_om()
{
    atc --input_format=NCHW --framework=1 --model=$model --output=lenet_quant --soc_version=Ascend310 &> atc.log
}

function compile_app()
{
    cd ../ascend310_infer/src || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/src/out/main --om_path=./lenet_quant.om --dataset_path=$data_path --acljson_path=../ascend310_infer/src/acl.json --device_id=$device_id &> infer.log
}

function cal_acc()
{
    python3.7 ../postprocess.py --result_path=./result_Files --label_path=$label_path  &> acc.log
}

air_to_om
if [ $? -ne 0 ]; then
    echo " air to om failed"
    exit 1
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
