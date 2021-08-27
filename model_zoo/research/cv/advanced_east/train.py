# Copyright 2021 Huawei Technologies Co., Ltd
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
#################train advanced_east on dataset########################
"""
import argparse
import datetime
import os
import time
import ast

from mindspore import context, Model
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn.optim import AdamWeightDecay
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from src.logger import get_logger
from src.config import config as cfg
from src.dataset import load_adEAST_dataset
from src.model import get_AdvancedEast_net

set_seed(1)


def parse_args():
    """parameters"""
    parser = argparse.ArgumentParser('mindspore adveast training')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend.')

    # network related
    parser.add_argument('--pre_trained', default=False, type=ast.literal_eval,
                        help='model_path, local pretrained model to load')

    # logging and checkpoint related
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=1, help='ckpt_interval')
    parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    args_opt = parser.parse_args()

    args_opt.epoch_num = cfg.epoch_num
    args_opt.batch_size = cfg.batch_size
    args_opt.ckpt_save_max = cfg.ckpt_save_max
    args_opt.data_dir = cfg.data_dir
    args_opt.mindsrecord_train_file = cfg.mindsrecord_train_file
    args_opt.mindsrecord_test_file = cfg.mindsrecord_test_file
    args_opt.last_model_name = cfg.last_model_name
    args_opt.saved_model_file_path = cfg.saved_model_file_path
    args_opt.ds_sink_mode = cfg.ds_sink_mode
    args_opt.is_train = True
    return args_opt


if __name__ == '__main__':
    args = parse_args()

    device_num = int(os.environ.get("DEVICE_NUM", 1))
    context.set_context(mode=context.GRAPH_MODE)
    workers = 32
    if args.is_distributed:
        if args.device_target == "Ascend":
            context.set_context(device_id=args.device_id, device_target=args.device_target)
            init()
        elif args.device_target == "GPU":
            context.set_context(device_target=args.device_target)
            init()
            args.rank = get_rank()

        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        context.set_context(device_id=args.device_id)

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)


    args.logger.save_args(args)
    # network
    args.logger.important_info('start create network')

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    # get network and init
    loss_net, train_net = get_AdvancedEast_net(args)
    loss_net.add_flags_recursive(fp32=True)
    train_net.set_train(False)
    # pre_trained
    if args.pre_trained:
        load_param_into_net(train_net, load_checkpoint(os.path.join(args.saved_model_file_path, args.last_model_name)))
    # define callbacks

    mindrecordfile256 = os.path.join(cfg.data_dir, cfg.mindsrecord_train_file_var + str(256) + '.mindrecord')

    train_dataset256, batch_num256 = load_adEAST_dataset(mindrecordfile256, batch_size=8,
                                                         device_num=device_num, rank_id=args.rank, is_training=True,
                                                         num_parallel_workers=workers)

    mindrecordfile384 = os.path.join(cfg.data_dir, cfg.mindsrecord_train_file_var + str(384) + '.mindrecord')
    train_dataset384, batch_num384 = load_adEAST_dataset(mindrecordfile384, batch_size=4,
                                                         device_num=device_num, rank_id=args.rank, is_training=True,
                                                         num_parallel_workers=workers)
    mindrecordfile448 = os.path.join(cfg.data_dir, cfg.mindsrecord_train_file_var + str(448) + '.mindrecord')
    train_dataset448, batch_num448 = load_adEAST_dataset(mindrecordfile448, batch_size=2,
                                                         device_num=device_num, rank_id=args.rank, is_training=True,
                                                         num_parallel_workers=workers)
    start = time.time()
    learning_rate = cfg.learning_rate_ascend if args.device_target == 'Ascend' else cfg.learning_rate_gpu
    decay = cfg.decay_ascend if args.device_target == 'Ascend' else cfg.decay_gpu
    # train model using the images resized to 256
    train_net.optimizer = AdamWeightDecay(train_net.weights, learning_rate=learning_rate
                                          , eps=1e-7, weight_decay=decay)
    model = Model(train_net)
    time_cb = TimeMonitor(data_size=batch_num256)
    loss_cb = LossMonitor(per_print_times=batch_num256)
    callbacks = []
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval * batch_num256,
                                   keep_checkpoint_max=args.ckpt_save_max)
    save_ckpt_path = args.saved_model_file_path
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                              directory=save_ckpt_path,
                              prefix='Epoch_A{}'.format(args.rank))
    if args.is_distributed & args.is_save_on_master:
        if args.rank == 0:
            callbacks.extend([time_cb, loss_cb, ckpt_cb])
        model.train(args.epoch_num, train_dataset=train_dataset256,
                    callbacks=callbacks, dataset_sink_mode=args.ds_sink_mode)
    else:
        callbacks.extend([time_cb, loss_cb, ckpt_cb])
        model.train(args.epoch_num, train_dataset=train_dataset256,
                    callbacks=callbacks, dataset_sink_mode=args.ds_sink_mode)
    print(time.time() - start)
    # train model using the images resized to 384
    model.optimizer = AdamWeightDecay(train_net.weights, learning_rate=learning_rate
                                      , eps=1e-7, weight_decay=decay)
    train_net.optimizer = AdamWeightDecay(train_net.weights, learning_rate=learning_rate
                                          , eps=1e-7, weight_decay=decay)
    model = Model(train_net)

    time_cb = TimeMonitor(data_size=batch_num384)
    loss_cb = LossMonitor(per_print_times=batch_num384)
    callbacks = []
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval * batch_num384,
                                   keep_checkpoint_max=args.ckpt_save_max)
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                              directory=save_ckpt_path,
                              prefix='Epoch_B{}'.format(args.rank))
    if args.is_distributed & args.is_save_on_master:
        if args.rank == 0:
            callbacks.extend([time_cb, loss_cb, ckpt_cb])
        model.train(args.epoch_num, train_dataset=train_dataset384,
                    callbacks=callbacks, dataset_sink_mode=args.ds_sink_mode)
    else:
        callbacks.extend([time_cb, loss_cb, ckpt_cb])
        model.train(args.epoch_num, train_dataset=train_dataset384,
                    callbacks=callbacks, dataset_sink_mode=args.ds_sink_mode)
    print(time.time() - start)

    # train model using the images resized to 448
    model.optimizer = AdamWeightDecay(train_net.weights, learning_rate=learning_rate
                                      , eps=1e-7, weight_decay=decay)
    train_net.optimizer = AdamWeightDecay(train_net.weights, learning_rate=learning_rate
                                          , eps=1e-7, weight_decay=decay)
    model = Model(train_net)

    time_cb = TimeMonitor(data_size=batch_num448)
    loss_cb = LossMonitor(per_print_times=batch_num448)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval * batch_num448,
                                   keep_checkpoint_max=args.ckpt_save_max)
    callbacks = []
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                              directory=save_ckpt_path,
                              prefix='Epoch_C{}'.format(args.rank))
    if args.is_distributed & args.is_save_on_master:
        if args.rank == 0:
            callbacks.extend([time_cb, loss_cb, ckpt_cb])
        model.train(args.epoch_num, train_dataset=train_dataset448,
                    callbacks=callbacks, dataset_sink_mode=args.ds_sink_mode)
    else:
        callbacks.extend([time_cb, loss_cb, ckpt_cb])
        model.train(args.epoch_num, train_dataset=train_dataset448,
                    callbacks=callbacks, dataset_sink_mode=args.ds_sink_mode)

    print(time.time() - start)
