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

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root python train.py --platform GPU \
   --model ResNet --max_epoch 600 --n_epochs 300 --device_num 8 --dataroot ./data/horse2zebra/  \
   --outputs_dir ./outputs > output.train.log 2>&1 &
