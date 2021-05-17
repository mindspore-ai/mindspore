# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import datetime
import argparse

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.common.tensor import Tensor
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size

from src.logger import get_logger
from src.dataset import create_BRDNetDataset
from src.models import BRDNet, BRDWithLossCell, TrainingWrapper


## Params
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_data', default='../dataset/waterloo5050step40colorimage/'
                    , type=str, help='path of train data')
parser.add_argument('--sigma', default=75, type=int, help='noise level')
parser.add_argument('--channel', default=3, type=int
                    , help='image channel, 3 for color, 1 for gray')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--use_modelarts', type=int, default=0
                    , help='1 for True, 0 for False; when set True, we should load dataset from obs with moxing')
parser.add_argument('--train_url', type=str, default='train_url/'
                    , help='needed by modelarts, but we donot use it because the name is ambiguous')
parser.add_argument('--data_url', type=str, default='data_url/'
                    , help='needed by modelarts, but we donot use it because the name is ambiguous')
parser.add_argument('--output_path', type=str, default='./output/'
                    , help='output_path,when use_modelarts is set True, it will be cache/output/')
parser.add_argument('--outer_path', type=str, default='s3://output/'
                    , help='obs path,to store e.g ckpt files ')

parser.add_argument('--device_target', type=str, default='Ascend'
                    , help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')
parser.add_argument('--ckpt_save_max', type=int, default=5
                    , help='Maximum number of checkpoint files can be saved. Default: 5.')

set_seed(1)
args = parser.parse_args()
save_dir = os.path.join(args.output_path, 'sigma_' + str(args.sigma) \
           + '_' + datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

if not args.use_modelarts and not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_lr(steps_per_epoch, max_epoch, init_lr):
    lr_each_step = []
    while max_epoch > 0:
        tem = min(30, max_epoch)
        for _ in range(steps_per_epoch*tem):
            lr_each_step.append(init_lr)
        max_epoch -= tem
        init_lr /= 10
    return lr_each_step


device_id = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                    device_target=args.device_target, save_graphs=False)

def train():

    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
            context.set_context(device_id=device_id)
        elif args.device_target == "GPU":
            init()

        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        if args.device_target == "Ascend":
            context.set_context(device_id=device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1


    args.logger = get_logger(save_dir, "BRDNet", args.rank)
    args.logger.save_args(args)

    if args.use_modelarts:
        import moxing as mox
        args.logger.info("copying train data from obs to cache....")
        mox.file.copy_parallel(args.train_data, 'cache/dataset') # the args.train_data must end by '/'
        args.logger.info("copying traindata finished....")
        args.train_data = 'cache/dataset/'  # the args.train_data must end by '/'

    dataset, batch_num = create_BRDNetDataset(args.train_data, args.sigma, \
                        args.channel, args.batch_size, args.group_size, args.rank, shuffle=True)

    args.steps_per_epoch = int(batch_num / args.batch_size / args.group_size)

    model = BRDNet(args.channel)

    if args.pretrain:
        if args.use_modelarts:
            import moxing as mox
            args.logger.info("copying pretrain model from obs to cache....")
            mox.file.copy_parallel(args.pretrain, 'cache/pretrain')
            args.logger.info("copying pretrain model finished....")
            args.pretrain = 'cache/pretrain/'+args.pretrain.split('/')[-1]

        args.logger.info('loading pre-trained model {} into network'.format(args.pretrain))
        load_param_into_net(model, load_checkpoint(args.pretrain))
        args.logger.info('loaded pre-trained model {} into network'.format(args.pretrain))


    model = BRDWithLossCell(model)
    model.set_train()

    lr_list = get_lr(args.steps_per_epoch, args.epoch, args.lr)
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=Tensor(lr_list, mindspore.float32))
    model = TrainingWrapper(model, optimizer)

    model = Model(model, amp_level="O3")

    # define callbacks
    if args.rank == 0:
        time_cb = TimeMonitor(data_size=batch_num)
        loss_cb = LossMonitor(per_print_times=10)
        callbacks = [time_cb, loss_cb]
    else:
        callbacks = None
    if args.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.steps_per_epoch*args.save_every,
                                       keep_checkpoint_max=args.ckpt_save_max)
        save_ckpt_path = os.path.join(save_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='channel_'+str(args.channel)+'_sigma_'+str(args.sigma)+'_rank_'+str(args.rank))
        callbacks.append(ckpt_cb)

    model.train(args.epoch, dataset, callbacks=callbacks, dataset_sink_mode=True)

    args.logger.info("training finished....")
    if args.use_modelarts:
        args.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(save_dir, args.outer_path)
        args.logger.info("copying finished....")

if __name__ == '__main__':

    train()
    args.logger.info('All task finished!')
