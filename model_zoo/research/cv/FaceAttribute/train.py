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
"""Face attribute train."""
import os
import time
import datetime
import argparse

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.communication.management import get_group_size, init, get_rank
from mindspore.nn import TrainOneStepCell
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, RunContext, _InternalCallbackParam, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from src.FaceAttribute.resnet18 import get_resnet18
from src.FaceAttribute.loss_factory import get_loss
from src.dataset_train import data_generator
from src.lrsche_factory import warmup_step
from src.logging import get_logger, AverageMeter
from src.config import config

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)


class BuildTrainNetwork(nn.Cell):
    '''Build train network.'''
    def __init__(self, my_network, my_criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = my_network
        self.criterion = my_criterion
        self.print = P.Print()

    def construct(self, input_data, label):
        logit0, logit1, logit2 = self.network(input_data)
        loss0 = self.criterion(logit0, logit1, logit2, label)
        return loss0


def parse_args():
    '''Argument for Face Attributes.'''
    parser = argparse.ArgumentParser('Face Attributes')

    parser.add_argument('--mindrecord_path', type=str, default='', help='dataset path, e.g. /home/data.mindrecord')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--local_rank', type=int, default=0, help='current rank to support distributed')
    parser.add_argument('--world_size', type=int, default=8, help='current process number to support distributed')

    arg, _ = parser.parse_known_args()

    return arg

if __name__ == "__main__":
    mindspore.set_seed(1)

    # logger
    args = parse_args()

    # init distributed
    if args.world_size != 1:
        init()
        args.local_rank = get_rank()
        args.world_size = get_group_size()

    args.per_batch_size = config.per_batch_size
    args.dst_h = config.dst_h
    args.dst_w = config.dst_w
    args.workers = config.workers
    args.attri_num = config.attri_num
    args.classes = config.classes
    args.backbone = config.backbone
    args.loss_scale = config.loss_scale
    args.flat_dim = config.flat_dim
    args.fc_dim = config.fc_dim
    args.lr = config.lr
    args.lr_scale = config.lr_scale
    args.lr_epochs = config.lr_epochs
    args.weight_decay = config.weight_decay
    args.momentum = config.momentum
    args.max_epoch = config.max_epoch
    args.warmup_epochs = config.warmup_epochs
    args.log_interval = config.log_interval
    args.ckpt_path = config.ckpt_path

    if args.world_size == 1:
        args.per_batch_size = 256
    else:
        args.lr = args.lr * 4.

    if args.world_size != 1:
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.world_size)

    # model and log save path
    args.outputs_dir = os.path.join(args.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.local_rank)
    loss_meter = AverageMeter('loss')

    # dataloader
    args.logger.info('start create dataloader')
    de_dataloader, steps_per_epoch, num_classes = data_generator(args)
    args.steps_per_epoch = steps_per_epoch
    args.num_classes = num_classes
    args.logger.info('end create dataloader')
    args.logger.save_args(args)

    # backbone and loss
    args.logger.important_info('start create network')
    create_network_start = time.time()
    network = get_resnet18(args)

    criterion = get_loss()

    # load pretrain model
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load model {} success'.format(args.pretrained))

    # optimizer and lr scheduler
    lr = warmup_step(args, gamma=0.1)
    opt = Momentum(params=network.trainable_params(),
                   learning_rate=lr,
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   loss_scale=args.loss_scale)

    train_net = BuildTrainNetwork(network, criterion)

    # mixed precision training
    criterion.add_flags_recursive(fp32=True)

    # package training process
    train_net = TrainOneStepCell(train_net, opt, sens=args.loss_scale)
    context.reset_auto_parallel_context()

    # checkpoint
    if args.local_rank == 0:
        ckpt_max_num = args.max_epoch
        train_config = CheckpointConfig(save_checkpoint_steps=args.steps_per_epoch, keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=train_config, directory=args.outputs_dir, prefix='{}'.format(args.local_rank))
        cb_params = _InternalCallbackParam()
        cb_params.train_network = train_net
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 0
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    train_net.set_train()
    t_end = time.time()
    t_epoch = time.time()
    old_progress = -1

    i = 0
    for _, (data, gt_classes) in enumerate(de_dataloader):

        data_tensor = Tensor(data, dtype=mstype.float32)
        gt_tensor = Tensor(gt_classes, dtype=mstype.int32)

        loss = train_net(data_tensor, gt_tensor)
        loss_meter.update(loss.asnumpy()[0])

        # save ckpt
        if args.local_rank == 0:
            cb_params.cur_step_num = i + 1
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if i % args.steps_per_epoch == 0 and args.local_rank == 0:
            cb_params.cur_epoch_num += 1

        # save Log
        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            args.logger.important_info('{}, graph compile time={:.2f}s'.format(args.backbone, time_for_graph_compile))

        if i % args.log_interval == 0 and args.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / args.steps_per_epoch)
            fps = args.per_batch_size * (i - old_progress) * args.world_size / time_used
            args.logger.info('epoch[{}], iter[{}], {}, {:.2f} imgs/sec'.format(epoch, i, loss_meter, fps))

            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if i % args.steps_per_epoch == 0 and args.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / args.steps_per_epoch)
            fps = args.per_batch_size * args.world_size * args.steps_per_epoch / epoch_time_used
            args.logger.info('=================================================')
            args.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            args.logger.info('=================================================')
            t_epoch = time.time()

        i += 1

    args.logger.info('--------- trains out ---------')
