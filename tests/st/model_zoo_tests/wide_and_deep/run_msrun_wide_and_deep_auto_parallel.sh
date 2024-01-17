#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
unset SLOG_PRINT_TO_STDOUT
CODE_DIR="./"
if [ -d ${BASE_PATH}/../../../../tests/models/official/recommend/wide_and_deep ]; then
    CODE_DIR=${BASE_PATH}/../../../../tests/models/official/recommend/wide_and_deep
elif [ -d ${BASE_PATH}/../../tests/models/official/recommend/wide_and_deep ]; then
    CODE_DIR=${BASE_PATH}/../../tests/models/official/recommend/wide_and_deep
else
     echo "[ERROR] code dir is not found"
fi
echo $CODE_DIR
rm -rf ${BASE_PATH}/wide_and_deep
cp -r ${CODE_DIR}  ${BASE_PATH}/wide_and_deep
cp -f ${BASE_PATH}/python_file_for_ci/train_and_test_multinpu_ci.py ${BASE_PATH}/wide_and_deep/train_and_test_multinpu_ci.py
cp -f ${BASE_PATH}/python_file_for_ci/__init__.py ${BASE_PATH}/wide_and_deep/__init__.py
cp -f ${BASE_PATH}/python_file_for_ci/config.py ${BASE_PATH}/wide_and_deep/src/config.py
cp -f ${BASE_PATH}/python_file_for_ci/callbacks.py ${BASE_PATH}/wide_and_deep/src/callbacks.py
cp -f ${BASE_PATH}/python_file_for_ci/datasets.py ${BASE_PATH}/wide_and_deep/src/datasets.py
cp -f ${BASE_PATH}/python_file_for_ci/wide_and_deep.py ${BASE_PATH}/wide_and_deep/src/wide_and_deep.py
source ${BASE_PATH}/env.sh
export PYTHONPATH=${BASE_PATH}/wide_and_deep/:$PYTHONPATH

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10969 --join=True --log_dir=./wide_and_deep_auto_parallel_logs pytest -s -v ./wide_and_deep/train_and_test_multinpu_ci.py
