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
"""export"""
import argparse
import numpy as np

from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.resnet import resnet50 as resnet
from src.config import config

parser = argparse.ArgumentParser(description='checkpoint export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_name", type=str, default="resnet_thor", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

    # define net
    net = resnet(class_num=config.class_num)
    net.add_flags_recursive(thor=False)

    # load checkpoint
    param_dict = load_checkpoint(args.ckpt_file)
    keys = list(param_dict.keys())
    for key in keys:
        if "damping" in key:
            param_dict.pop(key)
    load_param_into_net(net, param_dict)

    inputs = np.random.uniform(0.0, 1.0, size=[args.batch_size, 3, args.height, args.width]).astype(np.float32)
    export(net, Tensor(inputs), file_name=args.file_name, file_format=args.file_format)
