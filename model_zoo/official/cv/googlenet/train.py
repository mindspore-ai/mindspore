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
#################train googlent example on cifar10########################
python train.py
"""
import argparse
import os

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.config import cifar_cfg, imagenet_cfg
from src.dataset import create_dataset_cifar10, create_dataset_imagenet
from src.googlenet import GoogleNet
from src.CrossEntropySmooth import CrossEntropySmooth

set_seed(1)

def lr_steps_cifar10(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr_each_step.append(lr_max)
        elif i < decay_epoch_index[1]:
            lr_each_step.append(lr_max * 0.1)
        elif i < decay_epoch_index[2]:
            lr_each_step.append(lr_max * 0.01)
        else:
            lr_each_step.append(lr_max * 0.001)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def lr_steps_imagenet(_cfg, steps_per_epoch):
    """lr step for imagenet"""
    from src.lr_scheduler.warmup_step_lr import warmup_step_lr
    from src.lr_scheduler.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
    if _cfg.lr_scheduler == 'exponential':
        _lr = warmup_step_lr(_cfg.lr_init,
                             _cfg.lr_epochs,
                             steps_per_epoch,
                             _cfg.warmup_epochs,
                             _cfg.epoch_size,
                             gamma=_cfg.lr_gamma,
                            )
    elif _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         _cfg.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['imagenet', 'cifar10'],
                        help='dataset name.')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    args_opt = parser.parse_args()

    if args_opt.dataset_name == "cifar10":
        cfg = cifar_cfg
    elif args_opt.dataset_name == "imagenet":
        cfg = imagenet_cfg
    else:
        raise ValueError("Unsupported dataset.")

    # set context
    device_target = cfg.device_target

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    rank = 0
    if device_target == "Ascend":
        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
    elif device_target == "GPU":
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
    else:
        raise ValueError("Unsupported platform.")

    if args_opt.dataset_name == "cifar10":
        dataset = create_dataset_cifar10(cfg.data_path, 1)
    elif args_opt.dataset_name == "imagenet":
        dataset = create_dataset_imagenet(cfg.data_path, 1)
    else:
        raise ValueError("Unsupported dataset.")

    batch_num = dataset.get_dataset_size()

    net = GoogleNet(num_classes=cfg.num_classes)
    # Continue training if set pre_trained to be True
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(net, param_dict)

    loss_scale_manager = None
    if args_opt.dataset_name == 'cifar10':
        lr = lr_steps_cifar10(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size, steps_per_epoch=batch_num)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       learning_rate=Tensor(lr),
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    elif args_opt.dataset_name == 'imagenet':
        lr = lr_steps_imagenet(cfg, batch_num)


        def get_param_groups(network):
            """ get param groups """
            decay_params = []
            no_decay_params = []
            for x in network.trainable_params():
                parameter_name = x.name
                if parameter_name.endswith('.bias'):
                    # all bias not using weight decay
                    no_decay_params.append(x)
                elif parameter_name.endswith('.gamma'):
                    # bn weight bias not using weight decay, be carefully for now x not include BN
                    no_decay_params.append(x)
                elif parameter_name.endswith('.beta'):
                    # bn weight bias not using weight decay, be carefully for now x not include BN
                    no_decay_params.append(x)
                else:
                    decay_params.append(x)

            return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


        if cfg.is_dynamic_loss_scale:
            cfg.loss_scale = 1

        opt = Momentum(params=get_param_groups(net),
                       learning_rate=Tensor(lr),
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay,
                       loss_scale=cfg.loss_scale)
        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=loss_scale_manager)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_" + args_opt.dataset_name, directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("train success")
