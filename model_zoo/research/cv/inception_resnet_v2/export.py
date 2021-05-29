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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

# import mindspore as ms
from mindspore import Tensor, dtype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export, context

from src.config import config_ascend as config
from src.inception_resnet_v2 import Inception_resnet_v2

parser = argparse.ArgumentParser(description='inception_resnet_v2 export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument('--ckpt_file', type=str, required=True, help='inception_resnet_v2 ckpt file.')
parser.add_argument('--file_name', type=str, default='inception_resnet_v2', help='inception_resnet_v2 output air name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument('--width', type=int, default=299, help='input width')
parser.add_argument('--height', type=int, default=299, help='input height')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    net = Inception_resnet_v2(classes=config.num_classes)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([config.batch_size, 3, args.width, args.height]), dtype.float32)
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
