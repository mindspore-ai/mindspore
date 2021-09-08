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
"""train imagenet."""
import argparse
import math
import random

import numpy as np
import mindspore
from mindspore import Tensor, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn import SGD, RMSProp
from mindspore.context import ParallelMode
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.config import basic_config, dataset_config
from src.dataset import create_dataset
from src.efficientnet import efficientnet_b0
from src.loss import LabelSmoothingCrossEntropy

mindspore.common.set_seed(basic_config.random_seed)
random.seed(basic_config.random_seed)
np.random.seed(basic_config.random_seed)


def get_lr(base_lr, total_epochs, steps_per_epoch, decay_steps=1,
           decay_rate=0.9, warmup_steps=0., warmup_lr_init=0., global_epoch=0):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    global_steps = steps_per_epoch * global_epoch
    self_warmup_delta = ((base_lr - warmup_lr_init) /
                         warmup_steps) if warmup_steps > 0 else 0
    self_decay_rate = decay_rate if decay_rate < 1 else 1 / decay_rate
    for i in range(total_steps):
        steps = math.floor(i / steps_per_epoch)
        cond = 1 if (steps < warmup_steps) else 0
        warmup_lr = warmup_lr_init + steps * self_warmup_delta
        decay_nums = math.floor(steps / decay_steps)
        decay_rate = math.pow(self_decay_rate, decay_nums)
        decay_lr = base_lr * decay_rate
        lr = cond * warmup_lr + (1 - cond) * decay_lr
        lr_each_step.append(lr)
    lr_each_step = lr_each_step[global_steps:]
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


parser = argparse.ArgumentParser(description='Training configuration', add_help=False)
parser.add_argument('--data_path', type=str, metavar='DIR', required=True, help='path to dataset')
parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet', 'CIFAR10'],
                    help='ImageNet or CIFAR10')
parser.add_argument('--distributed', action='store_true', default=False)
parser.add_argument('--platform', type=str, default='GPU', choices=('GPU', 'CPU'), help='run platform')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')


def main():
    args, _ = parser.parse_known_args()
    print(args)

    rank_id, rank_size = 0, 1
    context.set_context(mode=context.GRAPH_MODE)

    if args.platform == "GPU":
        dataset_sink_mode = True
        context.set_context(device_target='GPU', enable_graph_kernel=True)
    elif args.platform == "CPU":
        dataset_sink_mode = False
        context.set_context(device_target='CPU')
    else:
        raise NotImplementedError("Training only supported for CPU and GPU.")

    if args.distributed:
        if args.platform == "GPU":
            init("nccl")
        else:
            raise NotImplementedError("Distributed Training only supported for GPU.")
        context.reset_auto_parallel_context()
        rank_id = get_rank()
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, device_num=rank_size)

    dataset_type = args.dataset.lower()
    cfg = dataset_config[dataset_type].cfg

    net = efficientnet_b0(num_classes=cfg.num_classes,
                          cfg=dataset_config[dataset_type],
                          drop_rate=cfg.drop,
                          drop_connect_rate=cfg.drop_connect,
                          global_pool=cfg.gp,
                          bn_tf=cfg.bn_tf,
                          )

    train_data_url = args.data_path
    train_dataset = create_dataset(
        dataset_type, train_data_url, cfg.batch_size, workers=cfg.workers, distributed=args.distributed)
    batches_per_epoch = train_dataset.get_dataset_size()
    print("Batches_per_epoch: ", batches_per_epoch)

    loss_cb = LossMonitor(per_print_times=1 if args.platform == "CPU" else batches_per_epoch)
    loss = LabelSmoothingCrossEntropy(smooth_factor=cfg.smoothing, num_classes=cfg.num_classes)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    loss_scale_manager = FixedLossScaleManager(
        cfg.loss_scale, drop_overflow_update=False)

    callbacks = [time_cb, loss_cb]

    if cfg.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(
            prefix=cfg.model, directory='./ckpt_' + str(rank_id) + '/', config=config_ck)
        callbacks += [ckpoint_cb]

    lr = Tensor(get_lr(base_lr=cfg.lr, total_epochs=cfg.epochs, steps_per_epoch=batches_per_epoch,
                       decay_steps=cfg.decay_epochs, decay_rate=cfg.decay_rate,
                       warmup_steps=cfg.warmup_epochs, warmup_lr_init=cfg.warmup_lr_init,
                       global_epoch=cfg.resume_start_epoch))
    if cfg.opt == 'sgd':
        optimizer = SGD(net.trainable_params(), learning_rate=lr, momentum=cfg.momentum,
                        weight_decay=cfg.weight_decay,
                        loss_scale=cfg.loss_scale
                        )
    elif cfg.opt == 'rmsprop':
        optimizer = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=cfg.weight_decay,
                            momentum=cfg.momentum, epsilon=cfg.opt_eps, loss_scale=cfg.loss_scale
                            )

    loss.add_flags_recursive(fp32=True, fp16=False)

    if args.resume:
        ckpt = load_checkpoint(args.resume)
        load_param_into_net(net, ckpt)

    model = Model(net, loss, optimizer,
                  loss_scale_manager=loss_scale_manager,
                  amp_level=cfg.amp_level
                  )

    if args.resume:
        real_epoch = cfg.epochs - cfg.resume_start_epoch
        model.train(real_epoch, train_dataset,
                    callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
    else:
        model.train(cfg.epochs, train_dataset,
                    callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    main()
