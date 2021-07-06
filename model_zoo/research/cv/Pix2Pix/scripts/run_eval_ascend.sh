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

echo "======================================================================================================================================="
echo "Please run the eval as: "
echo "python eval.py device_target device_id val_data_dir ckpt"
echo "for example: python eval.py --device_target Ascend --device_id 0 --val_data_dir ./facades/test --ckpt ./results/ckpt/Generator_200.ckpt"
echo "======================================================================================================================================="

python eval.py --device_target Ascend --device_id 0 --val_data_dir ./facades/test --ckpt ./results/ckpt/Generator_200.ckpt
