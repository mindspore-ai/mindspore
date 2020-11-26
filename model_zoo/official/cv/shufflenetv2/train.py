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
import ast
import os


import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.shufflenetv2 import ShuffleNetV2
from src.config import config_gpu as cfg
from src.dataset import create_dataset
from src.lr_generator import get_lr_basic
from src.CrossEntropySmooth import CrossEntropySmooth

set_seed(cfg.random_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--resume', type=str, default='', help='resume training with existed checkpoint')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help='distributed training')
    parser.add_argument('--platform', type=str, default='GPU', choices=('Ascend', 'GPU'), help='run platform')
    parser.add_argument('--model_size', type=str, default='1.0x', help='ShuffleNetV2 model size parameter')
    args_opt = parser.parse_args()

    if args_opt.platform != "GPU":
        raise ValueError("Only supported GPU training.")

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit():
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # init distributed
    if args_opt.is_distributed:
        init("nccl")
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=cfg.group_size,
                                          gradients_mean=True)
    else:
        cfg.rank = 0
        cfg.group_size = 1

    # dataloader
    dataset = create_dataset(args_opt.dataset_path, True, cfg.rank, cfg.group_size)
    batches_per_epoch = dataset.get_dataset_size()
    print("Batches Per Epoch: ", batches_per_epoch)
    # network
    net = ShuffleNetV2(n_class=cfg.num_classes, model_size=args_opt.model_size)

    # loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

    # learning rate schedule
    lr = get_lr_basic(lr_init=cfg.lr_init, total_epochs=cfg.epoch_size,
                      steps_per_epoch=batches_per_epoch, is_stair=True)
    lr = Tensor(lr)

    # optimizer
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': cfg.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    optimizer = Momentum(params=net.trainable_params(), learning_rate=Tensor(lr), momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay)
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}

    if args_opt.resume:
        ckpt = load_checkpoint(args_opt.resume)
        load_param_into_net(net, ckpt)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'})

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=cfg.keep_checkpoint_max)
    save_ckpt_path = os.path.join(cfg.ckpt_path, 'ckpt_' + str(cfg.rank) + '/')
    ckpoint_cb = ModelCheckpoint(prefix=f"shufflenet-rank{cfg.rank}", directory=save_ckpt_path, config=config_ck)
    if args_opt.is_distributed & cfg.is_save_on_master:
        if cfg.rank == 0:
            callbacks.append(ckpoint_cb)
        model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=True)
    else:
        callbacks.append(ckpoint_cb)
        model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=True)
    print("train success")
