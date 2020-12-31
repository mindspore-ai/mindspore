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
##############export checkpoint file into air , mindir and onnx models#################
python export.py --net squeezenet --dataset cifar10 --checkpoint_path squeezenet_cifar10-120_1562.ckpt
"""

import argparse
import numpy as np
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

parser = argparse.ArgumentParser(description='checkpoint export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument('--width', type=int, default=227, help='input width')
parser.add_argument('--height', type=int, default=227, help='input height')
parser.add_argument('--net', type=str, default='squeezenet', choices=['squeezenet', 'squeezenet_residual'],
                    help='Model.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'], help='Dataset.')
parser.add_argument("--file_name", type=str, default="squeezenet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

if args.net == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
if args.dataset == "cifar10":
    num_classes = 10
else:
    num_classes = 1000

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    net = squeezenet(num_classes=num_classes)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_data = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)
