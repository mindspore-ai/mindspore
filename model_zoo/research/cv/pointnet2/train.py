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
"""Training"""

import argparse
import ast
import os
import sys
import time

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.profiler import Profiler
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.dataset import DatasetGenerator
from src.lr_scheduler import MultiStepLR
from src.pointnet2 import PointNet2, NLLLoss
from src.provider import RandomInputDropout


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('MindSpore PointNet++ Training Configurations.')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')  # 24
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')  # 200
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')  # 0.001
    parser.add_argument('--optimizer', type=str, default='Adam', choices=('Adam', 'SGD'),
                        help='optimizer for training')  # Adam
    parser.add_argument('--data_path', type=str, default='../modelnet40_normal_resampled/', help='data path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--loss_per_epoch', type=int, default=5, help='times to print loss value per epoch')
    parser.add_argument('--save_dir', type=str, default='./save', help='save root')

    parser.add_argument('--use_normals', type=ast.literal_eval, default=False, help='use normals')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')

    parser.add_argument('--platform', type=str, default='Ascend', help='run platform')
    parser.add_argument('--enable_profiling', type=ast.literal_eval, default=False)

    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)
    parser.add_argument('--mox_freq', type=int, default=10, help='mox frequency')

    return parser.parse_known_args()[0]


def run_train():
    """run train"""
    args = parse_args()

    # INIT
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    rank_id = int(os.getenv('RANK_ID', '0'))

    if args.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        context.set_context(max_call_depth=2048)

        if device_num > 1:
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        raise ValueError("Unsupported platform.")

    if args.enable_modelarts:
        import moxing as mox

        local_data_url = "/cache/data"
        mox.file.copy_parallel(args.data_url, local_data_url)
        if args.pretrained_ckpt.endswith('.ckpt'):
            pretrained_ckpt_path = "/cache/pretrained_ckpt/pretrained.ckpt"
            mox.file.copy_parallel(args.pretrained_ckpt, pretrained_ckpt_path)
        local_train_url = "/cache/train_output"
        save_dir = local_train_url
        if rank_id == 0:
            mox.file.copy_parallel(os.path.join(args.train_url, 'log_train.txt'),
                                   os.path.join(save_dir, 'log_train.txt'))
            log_file = open(os.path.join(save_dir, 'log_train.txt'), 'w')
            sys.stdout = log_file
    else:
        local_data_url = args.data_path
        if args.pretrained_ckpt.endswith('.ckpt'):
            pretrained_ckpt_path = args.pretrained_ckpt
        local_train_url = args.save_dir
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if args.enable_profiling:
        profiler = Profiler()

    print(args)

    # DATA LOADING
    print('Load dataset ...')
    data_path = local_data_url

    num_workers = 4
    train_ds_generator = DatasetGenerator(root=data_path, args=args, split='train', process_data=args.process_data)
    if device_num > 1:
        train_ds = ds.GeneratorDataset(train_ds_generator, ["data", "label"], num_parallel_workers=num_workers,
                                       shuffle=True, shard_id=rank_id, num_shards=device_num)
    else:
        train_ds = ds.GeneratorDataset(train_ds_generator, ["data", "label"], num_parallel_workers=num_workers,
                                       shuffle=True)
    random_input_dropout = RandomInputDropout()
    train_ds = train_ds.batch(batch_size=args.batch_size, per_batch_map=random_input_dropout,
                              input_columns=["data", "label"], drop_remainder=True, num_parallel_workers=num_workers)

    steps_per_epoch = train_ds.get_dataset_size()

    # MODEL
    net = PointNet2(args.num_category, args.use_normals)

    # load checkpoint
    if args.pretrained_ckpt.endswith('.ckpt'):
        print("Load checkpoint: %s" % args.pretrained_ckpt)
        param_dict = load_checkpoint(pretrained_ckpt_path)
        load_param_into_net(net, param_dict)

    net_loss = NLLLoss()

    lr_epochs = list(range(20, 201, 20))
    lr_fun = MultiStepLR(args.learning_rate, lr_epochs, 0.7, steps_per_epoch, args.epoch)
    lr = lr_fun.get_lr()

    if args.optimizer == 'Adam':
        net_opt = nn.Adam(
            net.trainable_params(),
            learning_rate=Tensor(lr),
            beta1=0.9,
            beta2=0.999,
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        net_opt = nn.SGD(net.trainable_params(), learning_rate=args.learning_rate, momentum=0.9)

    model = Model(net, net_loss, net_opt)

    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=args.epoch)
    ckpt_cb = ModelCheckpoint(prefix="ckpt_pointnet2", directory=local_train_url, config=config_ck)

    loss_freq = max(steps_per_epoch // args.loss_per_epoch, 1)

    cb = []
    cb += [TimeMonitor()]
    cb += [LossMonitor(loss_freq)]
    if (not args.enable_modelarts) or (rank_id == 0):
        cb += [ckpt_cb]

    if args.enable_modelarts:
        from src.callbacks import MoxCallBack
        cb += [MoxCallBack(local_train_url, args.train_url, args.mox_freq)]

    # TRAINING
    net.set_train()
    print('Starting training ...')
    print('Time: ', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    if args.enable_modelarts:
        mox.file.copy_parallel(local_train_url, args.train_url)

    time_start = time.time()

    model.train(epoch=args.epoch, train_dataset=train_ds, callbacks=cb, dataset_sink_mode=True)

    # END
    print('End of training.')
    print('Total time cost: {} min'.format("%.2f" % ((time.time() - time_start) / 60)))

    if args.enable_profiling:
        profiler.analyse()

    if args.enable_modelarts and rank_id == 0:
        log_file.close()
        mox.file.copy_parallel(local_train_url, args.train_url)


if __name__ == '__main__':
    run_train()
