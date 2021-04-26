# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Face detection train."""
import os
import ast
import time
import datetime
import argparse
import numpy as np

from mindspore import context
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
from mindspore.common import dtype as mstype

from src.logging import get_logger
from src.data_preprocess import create_dataset
from src.config import config
from src.network_define import define_network

def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser('Yolov3 Face Detection')
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "CPU"),
                        help="run platform, support Ascend and CPU.")
    parser.add_argument('--mindrecord_path', type=str, default='', help='dataset path, e.g. /home/data.mindrecord')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--local_rank', type=int, default=0, help='current rank to support distributed')
    parser.add_argument('--world_size', type=int, default=8, help='current process number to support distributed')
    parser.add_argument("--use_loss_scale", type=ast.literal_eval, default=True,
                        help="Whether use dynamic loss scale, default is True.")

    args, _ = parser.parse_known_args()
    args.batch_size = config.batch_size
    args.warmup_lr = config.warmup_lr
    args.lr_rates = config.lr_rates
    if args.run_platform == "CPU":
        args.use_loss_scale = False
        args.world_size = 1
        args.local_rank = 0
    if args.world_size != 8:
        args.lr_steps = [i * 8 // args.world_size for i in config.lr_steps]
    else:
        args.lr_steps = config.lr_steps
    args.gamma = config.gamma
    args.weight_decay = config.weight_decay if args.world_size != 1 else 0.
    args.momentum = config.momentum
    args.max_epoch = config.max_epoch
    args.log_interval = config.log_interval
    args.ckpt_path = config.ckpt_path
    args.ckpt_interval = config.ckpt_interval
    args.outputs_dir = os.path.join(args.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    print('args.outputs_dir', args.outputs_dir)
    args.num_classes = config.num_classes
    args.anchors = config.anchors
    args.anchors_mask = config.anchors_mask
    args.num_anchors_list = [len(x) for x in args.anchors_mask]
    return args


def train(args):
    '''train'''
    print('=============yolov3 start trainging==================')
    devid = int(os.getenv('DEVICE_ID', '0')) if args.run_platform != 'CPU' else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.run_platform, save_graphs=False, device_id=devid)
    # init distributed
    if args.world_size != 1:
        init()
        args.local_rank = get_rank()
        args.world_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, device_num=args.world_size,
                                          gradients_mean=True)
    args.logger = get_logger(args.outputs_dir, args.local_rank)

    # dataloader
    ds = create_dataset(args)

    args.logger.important_info('start create network')
    create_network_start = time.time()

    train_net = define_network(args)

    # checkpoint
    ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
    train_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
    ckpt_cb = ModelCheckpoint(config=train_config, directory=args.outputs_dir, prefix='{}'.format(args.local_rank))
    cb_params = _InternalCallbackParam()
    cb_params.train_network = train_net
    cb_params.epoch_num = ckpt_max_num
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    train_net.set_train()
    t_end = time.time()
    t_epoch = time.time()
    old_progress = -1
    i = 0
    if args.use_loss_scale:
        scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_factor=2, scale_window=2000)
    for data in ds.create_tuple_iterator(output_numpy=True):
        batch_images = data[0]
        batch_labels = data[1]
        input_list = [Tensor(batch_images, mstype.float32)]
        for idx in range(2, 26):
            input_list.append(Tensor(data[idx], mstype.float32))
        if args.use_loss_scale:
            scaling_sens = Tensor(scale_manager.get_loss_scale(), dtype=mstype.float32)
            loss0, overflow, _ = train_net(*input_list, scaling_sens)
            overflow = np.all(overflow.asnumpy())
            if overflow:
                scale_manager.update_loss_scale(overflow)
            else:
                scale_manager.update_loss_scale(False)
            args.logger.info('rank[{}], iter[{}], loss[{}], overflow:{}, loss_scale:{}, lr:{}, batch_images:{}, '
                             'batch_labels:{}'.format(args.local_rank, i, loss0, overflow, scaling_sens, args.lr[i],
                                                      batch_images.shape, batch_labels.shape))
        else:
            loss0 = train_net(*input_list)
            args.logger.info('rank[{}], iter[{}], loss[{}], lr:{}, batch_images:{}, '
                             'batch_labels:{}'.format(args.local_rank, i, loss0, args.lr[i],
                                                      batch_images.shape, batch_labels.shape))
        # save ckpt
        cb_params.cur_step_num = i + 1  # current step number
        cb_params.batch_num = i + 2
        if args.local_rank == 0:
            ckpt_cb.step_end(run_context)

        # save Log
        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            args.logger.important_info('Yolov3, graph compile time={:.2f}s'.format(time_for_graph_compile))

        if i % args.steps_per_epoch == 0:
            cb_params.cur_epoch_num += 1

        if i % args.log_interval == 0 and args.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / args.steps_per_epoch)
            fps = args.batch_size * (i - old_progress) * args.world_size / time_used
            args.logger.info('epoch[{}], iter[{}], loss:[{}], {:.2f} imgs/sec'.format(epoch, i, loss0, fps))
            t_end = time.time()
            old_progress = i

        if i % args.steps_per_epoch == 0 and args.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / args.steps_per_epoch)
            fps = args.batch_size * args.world_size * args.steps_per_epoch / epoch_time_used
            args.logger.info('=================================================')
            args.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            args.logger.info('=================================================')
            t_epoch = time.time()

        i = i + 1

    args.logger.info('=============yolov3 training finished==================')


if __name__ == "__main__":
    arg = parse_args()
    train(arg)
