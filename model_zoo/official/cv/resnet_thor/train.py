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
"""train resnet."""
import os
import random
import argparse
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import dataset as de
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.model import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size

from src.model_thor import Model_Thor as Model
from src.resnet_thor import resnet50
from src.dataset import create_dataset
from src.crossentropy import CrossEntropy

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_num', type=int, default=1, help='Device num')
args_opt = parser.parse_args()

if args_opt.device_target == "Ascend":
    from src.thor import THOR
    from src.config import config
else:
    from src.thor import THOR_GPU as THOR
    from src.config import config_gpu as config

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)


def get_model_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch, decay_epochs=100):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        if epoch >= decay_epochs:
            lr_local = lr_local * 0.5
        if epoch >= decay_epochs + 1:
            lr_local = lr_local * 0.5
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def get_model_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)
    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now


if __name__ == '__main__':
    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if args_opt.run_distribute:
        # Ascend target
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            auto_parallel_context().set_all_reduce_fusion_split_indices([107], "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum2")
            auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum3")
            auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum4")
            auto_parallel_context().set_all_reduce_fusion_split_indices([27], "hccl_world_groupsum5")
            init()
        # GPU target
        else:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target)

    # define net
    step_size = dataset.get_dataset_size()
    damping = get_model_damping(0, config.damping_init, config.damping_decay, 90, step_size)
    lr = get_model_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    net = resnet50(class_num=config.class_num, damping=damping, loss_scale=config.loss_scale,
                   frequency=config.frequency, batch_size=config.batch_size)

    # define loss, model
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    opt = THOR(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), config.momentum,
               filter(lambda x: 'matrix_A' in x.name, net.get_parameters()),
               filter(lambda x: 'matrix_G' in x.name, net.get_parameters()),
               filter(lambda x: 'A_inv_max' in x.name, net.get_parameters()),
               filter(lambda x: 'G_inv_max' in x.name, net.get_parameters()),
               config.weight_decay, config.loss_scale)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if target == "Ascend":
        model = Model(net, loss_fn=loss, optimizer=opt, amp_level='O2', loss_scale_manager=loss_scale,
                      keep_batchnorm_fp32=False, metrics={'acc'}, frequency=config.frequency)
    else:
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=True, frequency=config.frequency)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size, dataset, callbacks=cb)
