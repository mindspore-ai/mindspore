#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

rm -rf output
mkdir output
python3 setup.py bdist_wheel
cp ./dist/mslite_bench*whl output
rm -rf dist build mslite_bench.egg-info
WHL_NAME=`find "$(pwd)" -name "*.whl"`
echo "build mslite_bench whl done, and save to ${WHL_NAME}"