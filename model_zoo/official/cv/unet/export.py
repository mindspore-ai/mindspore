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

from mindspore import Tensor, export, load_checkpoint, load_param_into_net, context

from src.unet.unet_model import UNet
from src.config import cfg_unet as cfg

parser = argparse.ArgumentParser(description='unet export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument('--width', type=int, default=572, help='input width')
parser.add_argument('--height', type=int, default=572, help='input height')
parser.add_argument("--file_name", type=str, default="unet", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    net = UNet(n_channels=cfg["num_channels"], n_classes=cfg["num_classes"])
    # return a parameter dict for model
    param_dict = load_checkpoint(args.ckpt_file)
    # load the parameter into net
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.ones([args.batch_size, cfg["num_channels"], args.height, args.width]).astype(np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)
