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
"""
import argparse
import datetime
import os

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset

from src.crossentropy import CrossEntropy
from src.warmup_step_lr import warmup_step_lr
from src.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
from src.warmup_step_lr import lr_steps
from src.utils.logging import get_logger
from src.utils.util import get_param_groups
from src.vgg import vgg16


set_seed(1)


def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore classification training')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=1, help='device id of GPU or Ascend. (Default: None)')

    # dataset related
    parser.add_argument('--dataset', type=str, choices=["cifar10", "imagenet2012"], default="cifar10")
    parser.add_argument('--data_path', type=str, default='', help='train data dir')

    # network related
    parser.add_argument('--pre_trained', default='', type=str, help='model_path, local pretrained model to load')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='decrease lr by a factor of exponential lr_scheduler')
    parser.add_argument('--eta_min', type=float, default=0., help='eta_min in cosine_annealing scheduler')
    parser.add_argument('--T_max', type=int, default=90, help='T-max in cosine_annealing scheduler')

    # logging and checkpoint related
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=5, help='ckpt_interval')
    parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    args_opt = parser.parse_args()
    args_opt = merge_args(args_opt, cloud_args)

    if args_opt.dataset == "cifar10":
        from src.config import cifar_cfg as cfg
    else:
        from src.config import imagenet_cfg as cfg

    args_opt.label_smooth = cfg.label_smooth
    args_opt.label_smooth_factor = cfg.label_smooth_factor
    args_opt.lr_scheduler = cfg.lr_scheduler
    args_opt.loss_scale = cfg.loss_scale
    args_opt.max_epoch = cfg.max_epoch
    args_opt.warmup_epochs = cfg.warmup_epochs
    args_opt.lr = cfg.lr
    args_opt.lr_init = cfg.lr_init
    args_opt.lr_max = cfg.lr_max
    args_opt.momentum = cfg.momentum
    args_opt.weight_decay = cfg.weight_decay
    args_opt.per_batch_size = cfg.batch_size
    args_opt.num_classes = cfg.num_classes
    args_opt.buffer_size = cfg.buffer_size
    args_opt.ckpt_save_max = cfg.keep_checkpoint_max
    args_opt.pad_mode = cfg.pad_mode
    args_opt.padding = cfg.padding
    args_opt.has_bias = cfg.has_bias
    args_opt.batch_norm = cfg.batch_norm
    args_opt.initialize_mode = cfg.initialize_mode
    args_opt.has_dropout = cfg.has_dropout

    args_opt.lr_epochs = list(map(int, cfg.lr_epochs.split(',')))
    args_opt.image_size = list(map(int, cfg.image_size.split(',')))

    return args_opt


def merge_args(args_opt, cloud_args):
    """dictionary"""
    args_dict = vars(args_opt)
    if isinstance(cloud_args, dict):
        for key_arg in cloud_args.keys():
            val = cloud_args[key_arg]
            if key_arg in args_dict and val:
                arg_type = type(args_dict[key_arg])
                if arg_type is not None:
                    val = arg_type(val)
                args_dict[key_arg] = val
    return args_opt


if __name__ == '__main__':
    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    device_num = int(os.environ.get("DEVICE_NUM", 1))
    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
            context.set_context(device_id=args.device_id)
        elif args.device_target == "GPU":
            init()

        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=[2, 18])
    else:
        if args.device_target == "Ascend":
            context.set_context(device_id=args.device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)

    if args.dataset == "cifar10":
        dataset = vgg_create_dataset(args.data_path, args.image_size, args.per_batch_size, args.rank, args.group_size)
    else:
        dataset = classification_dataset(args.data_path, args.image_size, args.per_batch_size,
                                         args.rank, args.group_size)

    batch_num = dataset.get_dataset_size()
    args.steps_per_epoch = dataset.get_dataset_size()
    args.logger.save_args(args)

    # network
    args.logger.important_info('start create network')

    # get network and init
    network = vgg16(args.num_classes, args)

    # pre_trained
    if args.pre_trained:
        load_param_into_net(network, load_checkpoint(args.pre_trained))

    # lr scheduler
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(args.lr,
                                        args.steps_per_epoch,
                                        args.warmup_epochs,
                                        args.max_epoch,
                                        args.T_max,
                                        args.eta_min)
    elif args.lr_scheduler == 'step':
        lr = lr_steps(0, lr_init=args.lr_init, lr_max=args.lr_max, warmup_epochs=args.warmup_epochs,
                      total_epochs=args.max_epoch, steps_per_epoch=batch_num)
    else:
        raise NotImplementedError(args.lr_scheduler)

    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   loss_scale=args.loss_scale)

    if args.dataset == "cifar10":
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(network, loss_fn=loss, optimizer=opt, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)
    else:
        if not args.label_smooth:
            args.label_smooth_factor = 0.0
        loss = CrossEntropy(smooth_factor=args.label_smooth_factor, num_classes=args.num_classes)

        loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
        model = Model(network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O2")

    # define callbacks
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor(per_print_times=batch_num)
    callbacks = [time_cb, loss_cb]
    if args.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval * args.steps_per_epoch,
                                       keep_checkpoint_max=args.ckpt_save_max)
        save_ckpt_path = os.path.join(args.outputs_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(args.rank))
        callbacks.append(ckpt_cb)

    model.train(args.max_epoch, dataset, callbacks=callbacks)
