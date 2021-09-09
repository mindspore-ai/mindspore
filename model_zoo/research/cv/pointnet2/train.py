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

import numpy as np

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.profiler import Profiler
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, _InternalCallbackParam
from mindspore.train.callback import RunContext
from src.dataset import DatasetGenerator
from src.lr_scheduler import MultiStepLR
from src.pointnet2 import PointNet2, NLLLoss
from src.provider import random_point_dropout, random_scale_point_cloud, shift_point_cloud


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('MindSpore PointNet++ Training Configurations.')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=('Adam', 'SGD'),
                        help='optimizer for training')
    parser.add_argument('--data_path', type=str, default='../modelnet40_normal_resampled/', help='data path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--loss_per_epoch', type=int, default=5, help='times to print loss value per epoch')

    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='./log', help='log root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', type=ast.literal_eval, default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')

    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')
    parser.add_argument('--modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)
    parser.add_argument('--mox_freq', type=int, default=10, help='mox frequency')
    parser.add_argument('--use_profiler', type=ast.literal_eval, default=False)

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = parse_args()
    # Init
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))

    if args.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        context.set_context(max_call_depth=2048)

        if device_num > 1:
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        raise ValueError("Unsupported platform.")

    if args.modelarts:
        import moxing as mox

        local_data_url = "/cache/data"
        mox.file.copy_parallel(args.data_url, local_data_url)
        if args.pretrained_ckpt.endswith('.ckpt'):
            pretrained_ckpt_path = "/cache/pretrained_ckpt/pretrained.ckpt"
            mox.file.copy_parallel(args.pretrained_ckpt, pretrained_ckpt_path)
        local_train_url = "/cache/train_output"
        log_dir = local_train_url
        mox.file.copy_parallel(os.path.join(args.train_url, 'log_train.txt'),
                               os.path.join(log_dir, 'log_train.txt'))
        log_file = open(os.path.join(log_dir, 'log_train.txt'), 'w')
        sys.stdout = log_file
    else:
        local_data_url = args.data_path
        if args.pretrained_ckpt.endswith('.ckpt'):
            pretrained_ckpt_path = args.pretrained_ckpt
        local_train_url = args.log_dir
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if args.use_profiler:
        profiler = Profiler()

    print(args)

    # DATA LOADING
    print('Load dataset ...')
    data_path = local_data_url

    train_dataset_generator = DatasetGenerator(root=data_path,
                                               args=args,
                                               split='train',
                                               process_data=args.process_data)
    if device_num > 1:
        train_dataset = ds.GeneratorDataset(train_dataset_generator,
                                            ["data", "label"],
                                            num_parallel_workers=8,
                                            shuffle=True, shard_id=device_id, num_shards=device_num)
    else:
        train_dataset = ds.GeneratorDataset(train_dataset_generator,
                                            ["data", "label"],
                                            num_parallel_workers=8,
                                            shuffle=True)
    train_dataset = train_dataset.batch(batch_size=args.batch_size,
                                        drop_remainder=True,
                                        num_parallel_workers=8)

    steps_per_epoch = train_dataset.get_dataset_size()

    # MODEL
    num_class = args.num_category

    net = PointNet2(num_class, args.use_normals)

    # load checkpoint
    if args.pretrained_ckpt.endswith('.ckpt'):
        print("Load checkpoint: %s" % args.pretrained_ckpt)
        param_dict = load_checkpoint(pretrained_ckpt_path)
        load_param_into_net(net, param_dict)

    net.set_train(True)

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

    net_with_loss = nn.WithLossCell(net, net_loss)
    train_net = nn.TrainOneStepCell(net_with_loss, net_opt)

    # checkpoint save
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch,
                                   keep_checkpoint_max=args.epoch)
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                              directory=local_train_url,
                              prefix="ckpt_pointnet2")

    cb_params = _InternalCallbackParam()
    cb_params.train_network = train_net
    cb_params.epoch_num = args.epoch
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)

    # TRAINING
    print('Starting training ...')
    ckpt_cb.begin(run_context)
    time_start = time.time()
    print('Time: ', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    if args.modelarts:
        mox.file.copy_parallel(local_train_url, args.train_url)

    loss_freq = steps_per_epoch // args.loss_per_epoch
    if loss_freq == 0:
        loss_freq = 1
    for epoch in range(1, args.epoch + 1):
        for batch_id, data in enumerate(train_dataset.create_dict_iterator()):
            t_0 = time.time()

            points = data['data']
            label = data['label']

            points = points.asnumpy()
            points = random_point_dropout(points)
            points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = Tensor(points)

            loss = train_net(points, label)

            if (not args.modelarts) or (device_id == 0):
                if (batch_id + 1) % loss_freq == 0:
                    print('epoch: ', epoch,
                          '\t| batch: ', batch_id + 1, '/', steps_per_epoch,
                          '\t| loss: ', "%.4f" % np.mean(loss.asnumpy()),
                          '\t| step_time: ', "%.4f" % (time.time() - t_0), ' s')

                # ckpt progress
                cb_params.cur_epoch_num = epoch
                cb_params.cur_step_num = batch_id + 1 + (epoch - 1) * steps_per_epoch
                cb_params.batch_num = batch_id + 1 + (epoch - 1) * steps_per_epoch
                ckpt_cb.step_end(run_context)

        if args.modelarts and epoch % args.mox_freq == 0:
            mox.file.copy_parallel(local_train_url, args.train_url)

    if (not args.modelarts) or (device_id == 0):
        print('End of training.')
        time_end = time.time()
        print('Total time cost: ', (time_end - time_start) / 60, ' min')

    # End
    if args.use_profiler:
        profiler.analyse()

    if args.modelarts:
        log_file.close()
        if device_id == 0:
            mox.file.copy_parallel(local_train_url, args.train_url)
