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
""" export MINDIR """
import argparse as arg
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, export, load_checkpoint
from src.ghostnet import ghostnet_1x
from src.config import config


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='SID export')
    parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend',
                        help='device where the code will be implemented')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'MINDIR'], default='MINDIR',
                        help='file format')
    parser.add_argument('--checkpoint_path', required=True, default=None, help='ckpt file path')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)

    ckpt_dir = args.checkpoint_path
    net = ghostnet_1x(num_classes=config.num_classes)
    load_checkpoint(ckpt_dir, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, 224, 224]), ms.float32)
    print(input_data.shape)
    export(net, input_data, file_name='ghost', file_format=args.file_format)
