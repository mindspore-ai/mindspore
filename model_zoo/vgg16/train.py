# Copyright 2020 Huawei Technologies Co., Ltd
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
#################train vgg16 example on cifar10########################
python train.py --data_path=$DATA_HOME --device_id=$DEVICE_ID
"""
import argparse
import os
import random

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from src.config import cifar_cfg as cfg
from src.dataset import vgg_create_dataset
from src.vgg import vgg16

random.seed(1)
np.random.seed(1)


def lr_steps(global_step, lr_init, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr_value = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr_value = float(lr_max) * base * base
            if lr_value < 0.0:
                lr_value = 0.0
        lr_each_step.append(lr_value)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cifar10 classification')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--data_path', type=str, default='./cifar', help='path where the dataset is saved')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--pre_trained', type=str, default=None, help='the pretrained checkpoint file path.')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)

    device_num = int(os.environ.get("DEVICE_NUM", 1))
    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True)
        init()

    dataset = vgg_create_dataset(args_opt.data_path, cfg.epoch_size)
    batch_num = dataset.get_dataset_size()

    net = vgg16(num_classes=cfg.num_classes)
    # pre_trained
    if args_opt.pre_trained:
        load_param_into_net(net, load_checkpoint(args_opt.pre_trained))

    lr = lr_steps(0, lr_init=cfg.lr_init, lr_max=cfg.lr_max, warmup_epochs=cfg.warmup_epochs,
                  total_epochs=cfg.epoch_size, steps_per_epoch=batch_num)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), cfg.momentum,
                   weight_decay=cfg.weight_decay)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpoint_cb = ModelCheckpoint(prefix="train_vgg_cifar10", directory="./", config=config_ck)
    loss_cb = LossMonitor()
    model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("train success")
