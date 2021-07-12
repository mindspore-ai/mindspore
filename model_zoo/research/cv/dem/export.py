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
######################## train DEMnet example ########################
train DEMnet
python train.py --data_path = /YourDataPath \
                --dataset = AwA or CUB \
                --train_mode = att, word or fusion
"""

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import export
from mindspore import Tensor

from src.set_parser import set_parser
from src.utils import acc_cfg, backbone_cfg, param_cfg, withlosscell_cfg

import numpy as np

if __name__ == "__main__":
    # Set graph mode, device id
    args = set_parser()
    context.set_context(mode=context.PYNATIVE_MODE, \
                        device_target=args.device_target, \
                        device_id=args.device_id)

    # Loading datasets and iterators
    if args.dataset == 'AwA':
        train_x, train_att, train_word, \
        test_x, test_att, test_word, \
        test_label, test_id = dataset_AwA(args.data_path)
    elif args.dataset == 'CUB':
        train_att, train_x, \
        test_x, test_att, \
        test_label, test_id = dataset_CUB(args.data_path)

    # Initialize parameters
    num = acc_cfg(args)
    lr, weight_decay, clip_param = param_cfg(args)
    save_ckpt = args.save_ckpt

    # Build network
    net = backbone_cfg(args)
    loss_fn = nn.MSELoss(reduction='mean')
    optim = nn.Adam(net.trainable_params(), lr, weight_decay)
    MyWithLossCell = withlosscell_cfg(args)
    loss_net = MyWithLossCell(net, loss_fn)
    train_net = MyTrainOneStepCell(loss_net, optim)

    print("============== Starting Exporting ==============")
    if args.train_mode == 'att':
        if args.dataset == 'AwA':
            input0 = Tensor(np.zeros([args.batch_size, 85]), mindspore.float32)
        elif args.dataset == 'CUB':
            input0 = Tensor(np.zeros([args.batch_size, 312]), mindspore.float32)
        export(net, input0, file_name=save_ckpt, file_format=args.file_format)
        print("Successfully convert to", args.file_format)

    elif args.train_mode == 'word':
        input0 = Tensor(np.zeros([args.batch_size, 1000]), mindspore.float32)
        export(net, input0, file_name=save_ckpt, file_format=args.file_format)
        print("Successfully convert to", args.file_format)

    elif args.train_mode == 'fusion':
        input1 = Tensor(np.zeros([args.batch_size, 85]), mindspore.float32)
        input2 = Tensor(np.zeros([args.batch_size, 1000]), mindspore.float32)
        export(net, input1, input2, file_name=save_ckpt, file_format=args.file_format)
        print("Successfully convert to", args.file_format)
