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
"""export AIR file."""
import argparse
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
import mindspore.nn as nn
import mindspore.ops as ops
from src.deeplab_v3plus import DeepLabV3Plus

parser = argparse.ArgumentParser(description='checkpoint export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--input_size", type=int, default=513, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="deeplabv3plus", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument('--model', type=str, default='DeepLabV3plus_s8', choices=['DeepLabV3plus_s16', 'DeepLabV3plus_s8'],
                    help='Select model structure (Default: DeepLabV3plus_s8)')
parser.add_argument('--num_classes', type=int, default=21, help='the number of classes (Default: 21)')
parser.add_argument("--input_format", type=str, choices=["NCHW", "NHWC"], default="NCHW",
                    help="NCHW or NHWC")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)


class BuildNetwork(nn.Cell):
    """
    create network for export air model
    the channels will be transposed to do infer
    """
    def __init__(self, net, input_format="NHWC"):
        super(BuildNetwork, self).__init__()
        self.network = net
        self.softmax = nn.Softmax(axis=1)
        self.format = input_format
        self.transpose = ops.Transpose()

    def construct(self, data):
        if self.format == "NHWC":
            data = self.transpose(data, (0, 3, 1, 2))
        output = self.network(data)
        output = self.softmax(output)
        return output


if __name__ == '__main__':
    if args.model == 'DeepLabV3plus_s16':
        network = DeepLabV3Plus('eval', args.num_classes, 16, True)
    else:
        network = DeepLabV3Plus('eval', args.num_classes, 8, True)
    network = BuildNetwork(network, args.input_format)
    network.set_train(False)
    param_dict = load_checkpoint(args.ckpt_file)

    # load the parameter into net
    load_param_into_net(network, param_dict)
    if args.input_format == "NHWC":
        input_data = Tensor(np.ones([args.batch_size, args.input_size, args.input_size, 3]).astype(np.float32))
    else:
        input_data = Tensor(np.ones([args.batch_size, 3, args.input_size, args.input_size]).astype(np.float32))
    export(network, input_data, file_name=args.file_name + args.input_format, file_format=args.file_format)
