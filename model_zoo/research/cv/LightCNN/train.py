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
""""train LightCNN."""

import os
import argparse

import mindspore.nn as nn
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.metrics import Accuracy, Top1CategoricalAccuracy, Top5CategoricalAccuracy

from src.lr_generator import get_lr
from src.config import lightcnn_cfg as cfg
from src.dataset import create_dataset
from src.lightcnn import lightCNN_9Layers


def parse_args():
    """parse train parameters."""
    parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--ckpt_path', type=str, default="", help='if is test, must provide\
                        path where the trained mat_files file')
    parser.add_argument('--run_distribute', type=int, default=0, help='0 -- run standalone, 1 -- run distribute')
    parser.add_argument('--resume', type=str, default='', help="resume model's checkpoint, please use \
                        checkpoint file name")

    args = parser.parse_args()

    return args


def main():
    """Main entrance for training"""
    args = parse_args()
    set_seed(1)

    # context parameters
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = 1
    rank_id = 0

    # init environment(distribute or not)
    if args.run_distribute:
        device_num = int(os.getenv('DEVICE_NUM'))
        rank_id = int(os.getenv("RANK_ID"))
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target, device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target, device_id=device_id)

    # define save checkpoint flag
    is_save_checkpoint = True
    if rank_id != 0:
        is_save_checkpoint = False

    # define dataset
    if args.run_distribute:
        ds_train = create_dataset(mode='Train',
                                  data_url=cfg.data_path,
                                  data_list=cfg.train_list,
                                  batch_size=cfg.batch_size,
                                  num_of_workers=8,
                                  is_distributed=True,
                                  group_size=get_group_size(),
                                  rank=get_rank(),
                                  seed=0
                                  )
    else:
        ds_train = create_dataset(mode='Train', data_url=cfg.data_path, data_list=cfg.train_list,
                                  batch_size=cfg.batch_size, num_of_workers=8)

    # define network
    network = lightCNN_9Layers(cfg.num_classes)

    # resume network
    if args.resume:
        if os.path.isfile(args.resume):
            net_parameters = load_checkpoint(args.resume)
            load_param_into_net(net_parameters, network)
        else:
            raise RuntimeError('No such file {}'.format(args.resume))

    # define dynamic learning rate
    steps_per_epoch = ds_train.get_dataset_size()
    bias_fc2_lr = get_lr(epoch_max=cfg.epochs, lr_base=cfg.lr * 20, steps_per_epoch=steps_per_epoch)
    bias_nfc2_lr = get_lr(epoch_max=cfg.epochs, lr_base=cfg.lr * 2, steps_per_epoch=steps_per_epoch)
    fc2_lr = get_lr(epoch_max=cfg.epochs, lr_base=cfg.lr * 10, steps_per_epoch=steps_per_epoch)
    nfc2_lr = get_lr(epoch_max=cfg.epochs, lr_base=cfg.lr, steps_per_epoch=steps_per_epoch)

    bias_fc2_lr = Tensor(bias_fc2_lr, mstype.float32)
    bias_nfc2_lr = Tensor(bias_nfc2_lr, mstype.float32)
    fc2_lr = Tensor(fc2_lr, mstype.float32)
    nfc2_lr = Tensor(nfc2_lr, mstype.float32)

    # define optimizer parameter
    params_dict = dict(network.parameters_and_names())
    bias_fc2 = []
    bias_nfc2 = []
    fc2 = []
    nfc2 = []
    for k, param in params_dict.items():
        if 'bias' in k:
            if 'fc2' in k:
                bias_fc2.append(param)
            else:
                bias_nfc2.append(param)
        else:
            if 'fc2' in k:
                fc2.append(param)
            else:
                nfc2.append(param)
    params = [
        {'params': bias_fc2, 'lr': bias_fc2_lr, 'weight_decay': 0},
        {'params': bias_nfc2, 'lr': bias_nfc2_lr, 'weight_decay': 0},
        {'params': fc2, 'lr': fc2_lr},
        {'params': nfc2},
    ]

    # define loss function
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # define optimizer
    net_opt = nn.SGD(params, nfc2_lr, cfg.momentum, weight_decay=cfg.weight_decay)

    # define model
    model = Model(network, net_loss, net_opt,
                  metrics={"Accuracy": Accuracy(),
                           "Top1": Top1CategoricalAccuracy(),
                           "Top5": Top5CategoricalAccuracy()},
                  amp_level="O3")

    # define callbacks
    callbacks = []
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callbacks.append(time_cb)
    callbacks.append(LossMonitor())
    if is_save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lightcnn", directory=args.ckpt_path, config=config_ck)
        callbacks.append(ckpoint_cb)

    print("============== Starting Training ==============")
    model.train(cfg['epochs'], ds_train, callbacks=callbacks)


if __name__ == '__main__':
    main()
