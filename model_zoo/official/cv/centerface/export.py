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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.centerface import CenterfaceMobilev2, CenterFaceWithNms
from src.config import ConfigCenterface

parser = argparse.ArgumentParser(description='centerface export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="centerface", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    config = ConfigCenterface()
    net = CenterfaceMobilev2()

    param_dict = load_checkpoint(args.ckpt_file)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.') or key.startswith('moment1.') or key.startswith('moment2.'):
            continue
        elif key.startswith('centerface_network.'):
            param_dict_new[key[19:]] = values
        else:
            param_dict_new[key] = values

    load_param_into_net(net, param_dict_new)
    net = CenterFaceWithNms(net)
    net.set_train(False)

    input_data = Tensor(np.zeros([args.batch_size, 3, config.input_h, config.input_w]), mindspore.float32)
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)
