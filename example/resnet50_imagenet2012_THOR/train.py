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
import random

import mindspore.dataset.engine as de
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import ParallelMode
from second_order.model_second_order import Model
from second_order.resnet import resnet50
from second_order.thor import THOR

import numpy as np
from config_imagenet import config
from crossentropy import CrossEntropy
from dataset_imagenet import create_dataset
from lr_generator import get_lr, warmup_cosine_annealing_lr

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')

args_opt = parser.parse_args()
device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True, device_id=device_id)
context.set_context(enable_task_sink=True)
context.set_context(enable_loop_sink=True)
context.set_context(enable_mem_reuse=True)


def get_second_order_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch):
    """get_second_order_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    print("learning_rate_is=====", lr_each_step)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def get_second_order_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_second_order_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)

    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    print("damping_is=========", damping_now)
    return damping_now


if __name__ == '__main__':
    if args_opt.do_eval:
        print("eval")
    else:
        if args_opt.run_distribute:
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True, parameter_broadcast=True)
            auto_parallel_context().set_all_reduce_fusion_split_indices([80], "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum3")
            auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum4")
            init()
        else:
            print(" ")

    epoch_size = config.epoch_size
    damping = get_second_order_damping(0, 0.03, 0.87, 50, 5004)
    net = resnet50(class_num=config.class_num, damping=damping, loss_scale=config.loss_scale,
                   frequency=config.frequency)

    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    if args_opt.do_train:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True,
                                 repeat_num=epoch_size, batch_size=config.batch_size)
        step_size = dataset.get_dataset_size()

        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        lr = Tensor(warmup_cosine_annealing_lr(0.035,
                                               step_size,
                                               config.warmup_epochs,
                                               50,
                                               config.T_max,
                                               config.eta_min))
        opt = THOR(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                   config.momentum, damping, config.frequency,
                   filter(lambda x: 'matrix_A' in x.name, net.get_parameters()),
                   filter(lambda x: 'matrix_G' in x.name, net.get_parameters()),
                   filter(lambda x: 'spatial_norm' in x.name, net.get_parameters()),
                   config.weight_decay, config.loss_scale)

        model = Model(net, loss_fn=loss, optimizer=opt, amp_level='O2', loss_scale_manager=loss_scale,
                      keep_batchnorm_fp32=False, metrics={'acc'}, frequency=config.frequency)

        time_cb = TimeMonitor(data_size=step_size)
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="resnet", directory=config.save_checkpoint_path, config=config_ck)
            cb += [ckpt_cb]

        model.train(epoch_size, dataset, callbacks=cb)
