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
python train.py
"""
import argparse
import os
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.train.model import Model, ParallelMode
from mindspore import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters

from src.dataset import create_dataset
from src.iresnet import iresnet100
from src.loss import PartialFC

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('--train_url', default='.', type=str,
                    help='output path')
parser.add_argument('--data_url', default='data path', type=str)
# Optimization options
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_classes', default=85742, type=int, metavar='N',
                    help='num of classes')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.08, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 16, 21],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Device options
parser.add_argument('--device_target', type=str,
                    default='Ascend', choices=['GPU', 'Ascend', 'CPU'])
parser.add_argument('--device_num', type=int, default=8)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--modelarts', type=bool, default=False)

args = parser.parse_args()


def lr_generator(lr_init, total_epochs, steps_per_epoch):
    '''lr_generator
    '''
    lr_each_step = []
    for i in range(total_epochs):
        if i in args.schedule:
            lr_init *= args.gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)


class MyNetWithLoss(nn.Cell):
    '''
    WithLossCell
    '''
    def __init__(self, backbone, cfg):
        super(MyNetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone.to_float(mstype.float16)
        self._loss_fn = PartialFC(num_classes=cfg.num_classes,
                                  world_size=cfg.device_num).to_float(mstype.float32)
        self.L2Norm = ops.L2Normalize(axis=1)

    def construct(self, data, label):
        out = self._backbone(data)
        loss = self._loss_fn(out, label)
        return loss


if __name__ == "__main__":
    train_epoch = args.epochs
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target, save_graphs=False)
    if args.device_num > 1:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
    else:
        context.set_context(device_id=args.device_id)
    if args.device_num > 1:
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          )
        cost_model_context.set_cost_model_context(device_memory_capacity=32.0 * 1024.0 * 1024.0 * 1024.0,
                                                  costmodel_gamma=0.001,
                                                  costmodel_beta=280.0)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        init()

    if args.modelarts:
        import moxing as mox

        mox.file.copy_parallel(
            src_url=args.data_url, dst_url='/cache/data_path_' + os.getenv('DEVICE_ID'))
        zip_command = "unzip -o -q /cache/data_path_" + os.getenv('DEVICE_ID') \
                      + "/MS1M.zip -d /cache/data_path_" + \
            os.getenv('DEVICE_ID')
        os.system(zip_command)
        train_dataset = create_dataset(dataset_path='/cache/data_path_' + os.getenv('DEVICE_ID') + '/MS1M/',
                                       do_train=True,
                                       repeat_num=1, batch_size=args.batch_size, target=target)
    else:
        train_dataset = create_dataset(dataset_path=args.data_url, do_train=True,
                                       repeat_num=1, batch_size=args.batch_size, target=target)
    step = train_dataset.get_dataset_size()
    lr = lr_generator(args.lr, train_epoch, steps_per_epoch=step)
    net = iresnet100()
    train_net = MyNetWithLoss(net, args)
    optimizer = nn.SGD(params=train_net.trainable_params(), learning_rate=lr / 512 * args.batch_size * args.device_num,
                       momentum=args.momentum, weight_decay=args.weight_decay)

    model = Model(train_net, optimizer=optimizer)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=60, keep_checkpoint_max=20)
    if args.modelarts:
        ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
                                  directory='/cache/train_output/')
    else:
        ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
                                  directory=args.train_url)
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [ckpt_cb, time_cb, loss_cb]
    if args.device_id == 0 or args.device_num == 1:
        model.train(train_epoch, train_dataset,
                    callbacks=cb, dataset_sink_mode=True)
    else:
        model.train(train_epoch, train_dataset, dataset_sink_mode=True)
    if args.modelarts:
        mox.file.copy_parallel(
            src_url='/cache/train_output', dst_url=args.train_url)
