# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import numpy as np

from mindspore import Tensor, export, load_checkpoint, load_param_into_net

from src.unet.unet_model import UNet

parser = argparse.ArgumentParser(description='Export ckpt to air')
parser.add_argument('--ckpt_file', type=str, default="ckpt_unet_medical_adam-1_600.ckpt",
                    help='The path of input ckpt file')
parser.add_argument('--air_file', type=str, default="unet_medical_adam-1_600.air", help='The path of output air file')
args = parser.parse_args()

net = UNet(n_channels=1, n_classes=2)
# return a parameter dict for model
param_dict = load_checkpoint(args.ckpt_file)
# load the parameter into net
load_param_into_net(net, param_dict)
input_data = np.random.uniform(0.0, 1.0, size=[1, 1, 572, 572]).astype(np.float32)
export(net, Tensor(input_data), file_name=args.air_file, file_format='AIR')
