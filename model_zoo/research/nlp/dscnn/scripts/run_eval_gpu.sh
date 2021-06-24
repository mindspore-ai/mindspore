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
# ===========================================================================

if [ $# != 2 ]
then
    echo "run as scripts/run_train_gpu.sh [EVAL_FEAT_DIR] [MODEL_DIR]"
    exit 1
fi

export DEVICE_ID=0
rm -rf eval_gpu
mkdir eval_gpu
cp -r ./src ./eval_gpu
cp -r ./scripts ./eval_gpu
cp ./*.py ./eval_gpu
cp ./*yaml ./eval_gpu
cd ./eval_gpu || exit
echo "start eval network"
python eval.py --model_dir $2 --device_target 'GPU' --eval_feat_dir $1 > eval.log 2>&1 &
cd ..
