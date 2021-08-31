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
"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import ast
import os

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.pointnet2 import PointNet2

parser = argparse.ArgumentParser(description='PointNet2 export')
parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False,
                    help="Run on modelArt, default is false.")
parser.add_argument('--data_url', default=None, help='Directory contains dataset.')
parser.add_argument('--train_url', default=None, help='Directory contains checkpoint file')
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file name.")
parser.add_argument("--batch_size", type=int, default=24, help="batch size")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')  # channels = 6 if true
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))
context.set_context(max_call_depth=2048)

if args.enable_modelarts:
    import moxing as mox

    local_data_url = "/cache/data"
    mox.file.copy_parallel(args.data_url, local_data_url)
    device_id = int(os.getenv('DEVICE_ID'))
    local_output_url = '/cache/ckpt' + str(device_id)
    mox.file.copy_parallel(src_url=os.path.join(args.train_url, args.ckpt_file),
                           dst_url=os.path.join(local_output_url, args.ckpt_file))
else:
    local_output_url = '.'

if __name__ == '__main__':

    net = PointNet2(args.num_category, args.use_normals)

    param_dict = load_checkpoint(os.path.join(local_output_url, args.ckpt_file))
    print('load ckpt')
    load_param_into_net(net, param_dict)
    print('load ckpt to net')
    net.set_train(False)
    input_arr = Tensor(np.ones([args.batch_size, 1024, 3]), mstype.float32)
    print('input')
    export(net, input_arr, file_name="PointNet2", file_format=args.file_format)
    if args.enable_modelarts:
        file_name = "PointNet2." + args.file_format.lower()
        mox.file.copy_parallel(src_url=file_name,
                               dst_url=os.path.join(args.train_url, file_name))
