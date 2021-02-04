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
"""train_imagenet."""
import argparse
import os

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.rmsprop import RMSProp
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.common import set_seed

from src.config import config_gpu, config_ascend, config_cpu
from src.dataset import create_dataset_imagenet, create_dataset_cifar10
from src.inception_v3 import InceptionV3
from src.lr_generator import get_lr
from src.loss import CrossEntropy

set_seed(1)

CFG_DICT = {
    "Ascend": config_ascend,
    "GPU": config_gpu,
    "CPU": config_cpu,
}

DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--resume', type=str, default='', help='resume training with existed checkpoint')
    parser.add_argument('--is_distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--platform', type=str, default='GPU', choices=('Ascend', 'GPU', 'CPU'), help='run platform')
    args_opt = parser.parse_args()

    cfg = CFG_DICT[args_opt.platform]
    create_dataset = DS_DICT[cfg.ds_type]

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # init distributed
    if args_opt.is_distributed:
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=cfg.group_size,
                                          gradients_mean=True)
    else:
        cfg.rank = 0
        cfg.group_size = 1

    # dataloader
    dataset = create_dataset(args_opt.dataset_path, True, cfg)
    batches_per_epoch = dataset.get_dataset_size()

    # network
    net = InceptionV3(num_classes=cfg.num_classes, dropout_keep_prob=cfg.dropout_keep_prob, has_bias=cfg.has_bias)

    # loss
    loss = CrossEntropy(smooth_factor=cfg.smooth_factor, num_classes=cfg.num_classes, factor=cfg.aux_factor)

    # learning rate schedule
    lr = get_lr(lr_init=cfg.lr_init, lr_end=cfg.lr_end, lr_max=cfg.lr_max, warmup_epochs=cfg.warmup_epochs,
                total_epochs=cfg.epoch_size, steps_per_epoch=batches_per_epoch, lr_decay_mode=cfg.decay_method)
    lr = Tensor(lr)

    # optimizer
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    if args_opt.platform == "Ascend":
        for param in net.trainable_params():
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                param.set_data(initializer(XavierUniform(), param.data.shape, param.data.dtype))
    group_params = [{'params': decayed_params, 'weight_decay': cfg.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    optimizer = RMSProp(group_params, lr, decay=0.9, weight_decay=cfg.weight_decay,
                        momentum=cfg.momentum, epsilon=cfg.opt_eps, loss_scale=cfg.loss_scale)
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}

    if args_opt.resume:
        ckpt = load_checkpoint(args_opt.resume)
        load_param_into_net(net, ckpt)
    if args_opt.platform == "Ascend":
        loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=cfg.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=cfg.amp_level)

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=cfg.keep_checkpoint_max)
    save_ckpt_path = os.path.join(cfg.ckpt_path, 'ckpt_' + str(cfg.rank) + '/')
    ckpoint_cb = ModelCheckpoint(prefix=f"inceptionv3-rank{cfg.rank}", directory=save_ckpt_path, config=config_ck)
    if args_opt.is_distributed & cfg.is_save_on_master:
        if cfg.rank == 0:
            callbacks.append(ckpoint_cb)
        model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=cfg.ds_sink_mode)
    else:
        callbacks.append(ckpoint_cb)
        model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=cfg.ds_sink_mode)
    print("train success")
