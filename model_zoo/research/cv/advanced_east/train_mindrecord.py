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


def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore adveast training')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: None)')

    # network related
    parser.add_argument('--pre_trained', default=False, type=bool, help='model_path, local pretrained model to load')
    parser.add_argument('--data_path', default='/disk1/ade/icpr/advanced-east-val_256.mindrecord', type=str)
    parser.add_argument('--pre_trained_ckpt', default='/disk1/adeast/scripts/1.ckpt', type=str)

    # logging and checkpoint related
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=1, help='ckpt_interval')
    parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    args_opt = parser.parse_args()

    args_opt.initial_epoch = cfg.initial_epoch
    args_opt.epoch_num = cfg.epoch_num
    args_opt.learning_rate = cfg.learning_rate
    args_opt.decay = cfg.decay
    args_opt.batch_size = cfg.batch_size
    args_opt.total_train_img = cfg.total_img * (1 - cfg.validation_split_ratio)
    args_opt.total_valid_img = cfg.total_img * cfg.validation_split_ratio
    args_opt.ckpt_save_max = cfg.ckpt_save_max
    args_opt.data_dir = cfg.data_dir
    args_opt.mindsrecord_train_file = cfg.mindsrecord_train_file
    args_opt.mindsrecord_test_file = cfg.mindsrecord_test_file
    args_opt.train_image_dir_name = cfg.train_image_dir_name
    args_opt.train_label_dir_name = cfg.train_label_dir_name
    args_opt.results_dir = cfg.results_dir
    args_opt.last_model_name = cfg.last_model_name
    args_opt.saved_model_file_path = cfg.saved_model_file_path
    return args_opt


if __name__ == '__main__':
    args = parse_args()
    context.set_context(device_target=args.device_target, device_id=args.device_id, mode=context.GRAPH_MODE)
    workers = 32
    device_num = 1
    if args.is_distributed:
        init()
        if args.device_target == "Ascend":
            device_num = int(os.environ.get("RANK_SIZE"))
            rank = int(os.environ.get("RANK_ID"))
            args.rank = args.device_id
        elif args.device_target == "GPU":
            context.set_context(device_target=args.device_target)
            device_num = get_group_size()
            args.rank = get_rank()
        args.group_size = device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=8, gradients_mean=True, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        context.set_context(device_id=args.device_id)

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)

    # dataset

    mindrecordfile = args.data_path

    train_dataset, batch_num = load_adEAST_dataset(mindrecordfile, batch_size=args.batch_size,
                                                   device_num=device_num, rank_id=args.rank, is_training=True,
                                                   num_parallel_workers=workers)

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
    loss_net, train_net = get_AdvancedEast_net()
    loss_net.add_flags_recursive(fp32=True)
    train_net.set_train(True)
    # pre_trained
    if args.pre_trained:
        load_param_into_net(train_net, load_checkpoint(args.pre_trained_ckpt))
    # define callbacks

    train_net.optimizer = AdamWeightDecay(train_net.weights, learning_rate=cfg.learning_rate
                                          , eps=1e-7, weight_decay=cfg.decay)
    model = Model(train_net)
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor(per_print_times=batch_num)
    callbacks = [time_cb, loss_cb]
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval * batch_num,
                                   keep_checkpoint_max=args.ckpt_save_max)
    save_ckpt_path = args.saved_model_file_path
    if args.rank_save_ckpt_flag:
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='Epoch_{}'.format(args.rank))
        callbacks.append(ckpt_cb)
    model.train(epoch=cfg.epoch_num, train_dataset=train_dataset, callbacks=callbacks, dataset_sink_mode=True)
