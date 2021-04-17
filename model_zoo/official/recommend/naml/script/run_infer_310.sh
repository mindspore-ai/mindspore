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

if [[ $# -lt 8 || $# -gt 9 ]]; then 
    echo "Usage: sh run_infer_310.sh [NEWS_MODEL] [USER_MODEL] [NEWS_DATASET_PATH] [USER_DATASET_PATH] [NEWS_ID_PATH] [USER_ID_PATH] [BROWSED_NEWS_PATH] [SOC_VERSION] [DEVICE_ID]
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

news_model=$(get_real_path $1)
user_model=$(get_real_path $2)
news_dataset_path=$(get_real_path $3)
user_dataset_path=$(get_real_path $4)
news_id_path=$(get_real_path $5)
user_id_path=$(get_real_path $6)
browsed_news_path=$(get_real_path $7)
soc_version=$8

if [ $# == 9 ]; then
    device_id=$9
elif [ $# == 8 ]; then
    if [ -z $device_id ]; then
        device_id=0
    else
        device_id=$device_id
    fi
fi

echo $news_model
echo $user_model
echo $news_dataset_path
echo $news_id_path
echo $user_id_path
echo $browsed_news_path
echo $soc_version
echo $device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export PATH=$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function air_to_om()
{
    atc --framework=1 --model=$news_model --output=news_encoder --soc_version=$soc_version
    atc --framework=1 --model=$user_model --output=user_encoder --soc_version=$soc_version
}

function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
     if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/out/main --news_om_path news_encoder.om  --user_om_path user_encoder.om --news_dataset_path $news_dataset_path --user_dataset_path $user_dataset_path --newsid_data_path $news_id_path --userid_data_path $user_id_path --browsed_news_path $browsed_news_path &> infer.log
    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    python3 ../postprocess.py --result_path=./result_Files --label_path=$browsed_news_path/02_labels_data  &> acc.log
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}

air_to_om
compile_app
infer
cal_acc
