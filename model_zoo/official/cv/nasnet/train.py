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
import os

from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.rmsprop import RMSProp
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.common import dtype as mstype

from src.config import nasnet_a_mobile_config_gpu as cfg
from src.dataset import create_dataset
from src.nasnet_a_mobile import NASNetAMobileWithLoss
from src.lr_generator import get_lr


set_seed(cfg.random_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--resume', type=str, default='', help='resume training with existed checkpoint')
    parser.add_argument('--is_distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--platform', type=str, default='GPU', choices=('Ascend', 'GPU'), help='run platform')
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
    dataset = create_dataset(args_opt.dataset_path, cfg, True)
    batches_per_epoch = dataset.get_dataset_size()

    # network
    net_with_loss = NASNetAMobileWithLoss(cfg)
    if args_opt.resume:
        ckpt = load_checkpoint(args_opt.resume)
        load_param_into_net(net_with_loss, ckpt)

    # learning rate schedule
    lr = get_lr(lr_init=cfg.lr_init, lr_decay_rate=cfg.lr_decay_rate,
                num_epoch_per_decay=cfg.num_epoch_per_decay, total_epochs=cfg.epoch_size,
                steps_per_epoch=batches_per_epoch, is_stair=True)
    if args_opt.resume:
        name_dir = os.path.basename(args_opt.resume)
        name, ext = name_dir.split(".")
        split_result = name.split("_")
        resume = split_result[-2].split("-")
        resume_epoch = int(resume[-1])
        step_num_in_epoch = int(split_result[-1])
        assert step_num_in_epoch == dataset.get_dataset_size()\
        , "This script only supports resuming at the end of epoch"
        lr = lr[(dataset.get_dataset_size() * (resume_epoch - 1) + step_num_in_epoch):]
    lr = Tensor(lr, mstype.float32)

    # optimizer
    decayed_params = []
    no_decayed_params = []
    for param in net_with_loss.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    group_params = [{'params': decayed_params, 'weight_decay': cfg.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net_with_loss.trainable_params()}]
    optimizer = RMSProp(group_params, lr, decay=cfg.rmsprop_decay, weight_decay=cfg.weight_decay,
                        momentum=cfg.momentum, epsilon=cfg.opt_eps, loss_scale=cfg.loss_scale)

    # high performance
    net_with_loss.set_train()
    model = Model(net_with_loss, optimizer=optimizer)

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=cfg.keep_checkpoint_max)
    save_ckpt_path = os.path.join(cfg.ckpt_path, 'ckpt_' + str(cfg.rank) + '/')
    ckpoint_cb = ModelCheckpoint(prefix=f"nasnet-a-mobile-rank{cfg.rank}", directory=save_ckpt_path, config=config_ck)
    if args_opt.is_distributed & cfg.is_save_on_master:
        if cfg.rank == 0:
            callbacks.append(ckpoint_cb)
        if args_opt.resume:
            model.train(cfg.epoch_size - resume_epoch, dataset, callbacks=callbacks, dataset_sink_mode=True)
        else:
            model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=True)
    else:
        callbacks.append(ckpoint_cb)
        if args_opt.resume:
            model.train(cfg.epoch_size - resume_epoch, dataset, callbacks=callbacks, dataset_sink_mode=True)
        else:
            model.train(cfg.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=True)
    print("train success")
