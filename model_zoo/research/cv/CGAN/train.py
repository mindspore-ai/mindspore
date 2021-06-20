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
"""train"""
import os
import time
import argparse
import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_group_size
import mindspore.ops as ops
from src.dataset import create_dataset
from src.ckpt_util import save_ckpt
from src.model import Generator, Discriminator
from src.cell import GenWithLossCell, DisWithLossCell, TrainOneStepCell


def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument("--distribute", type=bool, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='ckpt', help='checkpoint dir of CGAN')
    parser.add_argument('--dataset', type=str, default='data/MNIST_Data/train',
                        help='dataset dir (default data/MNISt_Data/train)')
    args = parser.parse_args()

    # if not exists 'imgs4', 'gif' or 'ckpt_dir', make it
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    # deal with the distribute analyze problem
    if args.distribute:
        device_id = args.device_id
        context.set_context(save_graphs=False,
                            device_id=device_id,
                            device_target="Ascend",
                            mode=context.GRAPH_MODE)
        init()
        args.device_num = get_group_size()
        context.set_auto_parallel_context(gradients_mean=True,
                                          device_num=args.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        device_id = args.device_id
        args.device_num = 1
        context.set_context(save_graphs=False,
                            mode=context.GRAPH_MODE,
                            device_target="Ascend")
        context.set_context(device_id=device_id)
    return args


def main():
    # before training, we should set some arguments
    args = preLauch()

    # training argument
    batch_size = 128
    input_dim = 100
    epoch_start = 0
    epoch_end = 51
    lr = 0.001

    dataset = create_dataset(args.dataset,
                             flatten_size=28 * 28,
                             batch_size=batch_size,
                             num_parallel_workers=args.device_num)

    # create G Cell & D Cell
    netG = Generator(input_dim)
    netD = Discriminator(batch_size)
    # create WithLossCell
    netG_with_loss = GenWithLossCell(netG, netD)
    netD_with_loss = DisWithLossCell(netG, netD)
    # create optimizer cell
    optimizerG = nn.Adam(netG.trainable_params(), lr)
    optimizerD = nn.Adam(netD.trainable_params(), lr)

    net_train = TrainOneStepCell(netG_with_loss,
                                 netD_with_loss,
                                 optimizerG,
                                 optimizerD)

    netG.set_train()
    netD.set_train()

    # latent_code_eval = Tensor(np.random.randn(
    #     200, input_dim), dtype=mstype.float32)

    # label_eval = np.zeros((200, 10))
    # for i in range(200):
    #     j = i // 20
    #     label_eval[i][j] = 1
    # label_eval = Tensor(label_eval, dtype=mstype.float32)

    data_size = dataset.get_dataset_size()
    print("data-size", data_size)
    print("=========== start training ===========")
    for epoch in range(epoch_start, epoch_end):
        step = 0
        start = time.time()
        for data in dataset:
            img = data[0]
            label = data[1]
            img = ops.Reshape()(img, (batch_size, 1, 28, 28))
            latent_code = Tensor(np.random.randn(
                batch_size, input_dim), dtype=mstype.float32)
            dout, gout = net_train(img, latent_code, label)
            step += 1

            if step % data_size == 0:
                end = time.time()
                pref = (end-start)*1000 / data_size
                print("epoch {}, {:.3f} ms per step, d_loss is {:.4f}, g_loss is {:.4f}".format(epoch,
                                                                                                pref, dout.asnumpy(),
                                                                                                gout.asnumpy()))

    save_ckpt(args, netG, netD, epoch)
    print("===========training success================")

if __name__ == '__main__':
    main()
