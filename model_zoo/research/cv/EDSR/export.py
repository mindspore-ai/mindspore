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
"""export net together with checkpoint into air/mindir/onnx models"""
import os
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
import src.model as edsr

parser = argparse.ArgumentParser(description='edsr export')
parser.add_argument("--ckpt_path", type=str, required=True, help="path of checkpoint file")
parser.add_argument("--file_name", type=str, default="edsr", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['MINDIR', 'AIR', 'ONNX'], help="file format")

parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--n_resblocks', type=int, default=32, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=256, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
parser.add_argument('--task_id', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=1)

args1 = parser.parse_args()
args1.scale = [int(x) for x in args1.scale.split("+")]
for arg in vars(args1):
    if vars(args1)[arg] == 'True':
        vars(args1)[arg] = True
    elif vars(args1)[arg] == 'False':
        vars(args1)[arg] = False

def run_export(args):
    """run_export"""
    device_id = int(os.getenv("DEVICE_ID", '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    net = edsr.EDSR(args)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    print('load mindspore net and checkpoint successfully.')
    inputs = Tensor(np.zeros([args.batch_size, 3, 678, 1020], np.float32))
    export(net, inputs, file_name=args.file_name, file_format=args.file_format)
    print('export successfully!')


if __name__ == "__main__":
    run_export(args1)
