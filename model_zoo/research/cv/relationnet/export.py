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

import os
import argparse
import numpy as np
from mindspore.common import dtype as mstype
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.config import relationnet_cfg as cfg
from src.relationnet import Encoder_Relation

parser = argparse.ArgumentParser(description='MindSpore RelationNet Example')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="relationnet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":

    # define fusion network
    network = Encoder_Relation(cfg.feature_dim, cfg.relation_dim)
    # load network checkpoint
    if os.path.exists(args.ckpt_file):
        param_dict = load_checkpoint(args.ckpt_file)
        load_param_into_net(network, param_dict)
        print('successfully load parameters')
    else:
        print('Load params Error')

    # export network
    inputs = Tensor(np.ones([100, 1, cfg.image_height, cfg.image_width]), mstype.float32)
    export(network, inputs, file_name=args.file_name, file_format=args.file_format)
    print('model is exported')
