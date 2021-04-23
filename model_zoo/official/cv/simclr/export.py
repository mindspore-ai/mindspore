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
"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

from src.simclr_model import SimCLR
from src.resnet import resnet50 as resnet

parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10'],
                    help='Dataset, Currently only cifar10 is supported.')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend'],
                    help='Device target, Currently only Ascend is supported.')
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="simclr", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
if args_opt.device_target == "Ascend":
    context.set_context(device_id=args_opt.device_id)

if __name__ == '__main__':
    if args_opt.dataset_name == 'cifar10':
        width_multiplier = 1
        cifar_stem = True
        projection_dimension = 128
        image_height = 32
        image_width = 32
    else:
        raise ValueError("dataset is not support.")

    base_net = resnet(1, width_multiplier=width_multiplier, cifar_stem=cifar_stem)
    net = SimCLR(base_net, projection_dimension, base_net.end_point.in_channels)

    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args_opt.batch_size, 3, image_height, image_width]), ms.float32)
    export(net, input_arr, file_name=args_opt.file_name, file_format=args_opt.file_format)
