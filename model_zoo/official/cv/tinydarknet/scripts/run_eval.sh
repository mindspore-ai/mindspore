#!/usr/bin/env bash
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

abs_path=$(readlink -f "$0")
cur_path=$(dirname $abs_path)
cd $cur_path

rm -rf ./eval
mkdir ./eval
cp -r ../src ./eval
cp ../eval.py ./eval
cp -r ../config ./eval
cd ./eval || exit
env >env.log
python ./eval.py > ./eval.log 2>&1 &
cd ..
