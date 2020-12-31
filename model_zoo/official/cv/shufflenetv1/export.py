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
"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.config import config as cfg
from src.shufflenetv1 import ShuffleNetV1

parser = argparse.ArgumentParser(description='ShuffleNetV1 export')
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="checkpoint file path.")
parser.add_argument("--file_name", type=str, default="shufflenetv1", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument('--model_size', type=str, default='2.0x', choices=['2.0x', '1.5x', '1.0x', '0.5x'],
                    help='shufflenetv1 model size')

args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':

    net = ShuffleNetV1(model_size=args.model_size)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    image_height, image_width = (224, 224)
    input_arr = Tensor(np.ones([cfg.batch_size, 3, image_height, image_width]), ms.float32)
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
