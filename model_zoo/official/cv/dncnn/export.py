#!/usr/bin/env python3
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

import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.model import DnCNN

parser = argparse.ArgumentParser(description='DnCNN')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--image_height", type=int, default=256, help="image_height")
parser.add_argument("--image_width", type=int, default=256, help="image_width")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="DnCNN", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument('--model_type', type=str, default='DnCNN-S', \
                    choices=['DnCNN-S', 'DnCNN-B', 'DnCNN-3'], help='type of DnCNN')

args = parser.parse_args()


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if args.model_type == 'DnCNN-S':
        network = DnCNN(1, num_of_layers=17)
    elif args.model_type == 'DnCNN-3' or args.model_type == 'DnCNN-B':
        network = DnCNN(1, num_of_layers=20)
    else:
        print("wrong model type")
        exit()

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    input_arr = Tensor(np.ones([args.batch_size, 1, args.image_height, args.image_width]), ms.float32)
    export(network, input_arr, file_name=args.file_name, file_format=args.file_format)
