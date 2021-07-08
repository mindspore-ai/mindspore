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

rm -rf evaluation
mkdir evaluation
cp ./*.py ./evaluation
cp ./*.yaml ./evaluation
cp -r ./src ./evaluation
cd ./evaluation || exit

DATA_DIR=$1
CKPT_DIR=$2
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config_cpu.yaml"

echo "start evaluation"

python eval.py --config_path=$CONFIG_FILE --dataset_path=$DATA_DIR --checkpoint_path=$CKPT_DIR \
--platform='CPU' > eval.log 2>&1 &
