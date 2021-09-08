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
import os
import ast
import argparse
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.wide_resnet import wideresnet

parser = argparse.ArgumentParser(description='WideResNet export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument('--data_url', default=None, help='Directory contains cifar10 dataset.')
parser.add_argument('--train_url', default=None, help='Directory contains checkpoint file')
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file name.")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if args.run_modelart:
    import moxing as mox
    device_id = int(os.getenv('DEVICE_ID'))
    local_output_url = '/cache/ckpt' + str(device_id)
    mox.file.copy_parallel(src_url=os.path.join(args.train_url, args.ckpt_file),
                           dst_url=os.path.join(local_output_url, args.ckpt_file))

if __name__ == '__main__':

    net = wideresnet()

    print('load ckpt')
    param_dict = load_checkpoint(args.ckpt_file)
    print('load ckpt to net')
    load_param_into_net(net, param_dict)
    net.set_train(False)
    input_arr = Tensor(np.ones([args.batch_size, 3, 32, 32]), mstype.float32)
    print('input')
    export(net, input_arr, file_name="WideResNet", file_format=args.file_format)
    if args.run_modelart:
        file_name = "WideResNet." + args.file_format.lower()
        mox.file.copy_parallel(src_url=file_name,
                               dst_url=os.path.join(args.train_url, file_name))
