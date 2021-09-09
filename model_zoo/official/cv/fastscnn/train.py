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
'''train.py'''
import os
import math
import argparse
import datetime

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore import FixedLossScaleManager
from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision.py_transforms import ToTensor, Normalize
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint

from src.logger import get_logger
from src.lr_scheduler import LRScheduler
from src.dataloader import create_CitySegmentation
from src.fast_scnn import FastSCNN, FastSCNNWithLossCell
from src.util import SegmentationMetric, EvalCallBack, apply_eval, TempLoss


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on mindspore')
    parser.add_argument('--dataset', type=str, default='/data/dataset/citys/',
                        help='dataset name (default: /data/dataset/citys/)')
    parser.add_argument('--base_size', type=int, default=1024, help='base image size')
    parser.add_argument('--crop_size', type=int, default=(768, 768), help='crop image size')
    parser.add_argument('--train_split', type=str, default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--aux', action='store_true', default=True, help='Auxiliary loss')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--save_every', type=int, default=1, metavar='N',
                        help='save ckpt every N epoch')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resuming file name')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='base learning rate (default: 0.045)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=4e-5, metavar='M',
                        help='w-decay (default: 4e-5)')

    parser.add_argument('--eval_while_train', type=int, default=1, help='eval while training')
    parser.add_argument('--eval_steps', type=int, default=10, help='each N epochs we eval')
    parser.add_argument('--eval_start_epoch', type=int, default=850, help='eval_start_epoch')
    parser.add_argument('--use_modelarts', type=int, default=0,
                        help='when set True, we should load dataset from obs with moxing')
    parser.add_argument('--train_url', type=str, default='train_url/',
                        help='needed by modelarts, but we donot use it because the name is ambiguous')
    parser.add_argument('--data_url', type=str, default='data_url/',
                        help='needed by modelarts, but we donot use it because the name is ambiguous')
    parser.add_argument('--output_path', type=str, default='./outputs/',
                        help='output_path,when use_modelarts is set True, it will be cache/output/')
    parser.add_argument('--outer_path', type=str, default='s3://output/',
                        help='obs path,to store e.g ckpt files ')

    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--is_save_on_master', type=int, default=1,
                        help='save ckpt on master or all rank')
    parser.add_argument('--ckpt_save_max', type=int, default=800,
                        help='Maximum number of checkpoint files can be saved. Default: 800')
    # the parser
    args_ = parser.parse_args()
    return args_

args = parse_args()
set_seed(1)
device_id = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
save_dir = os.path.join(args.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

def train():
    '''train'''
    if args.is_distributed:
        assert args.device_target == "Ascend"
        context.set_context(device_id=device_id)
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        if args.device_target in ["Ascend", "GPU"]:
            context.set_context(device_id=device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    args.logger = get_logger(save_dir, "Fast_SCNN", args.rank)
    args.logger.save_args(args)

    # image transform
    input_transform = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.use_modelarts:
        import moxing as mox
        args.logger.info("copying dataset from obs to cache....")
        mox.file.copy_parallel(args.dataset, 'cache/dataset')
        args.logger.info("copying dataset finished....")
        args.dataset = 'cache/dataset/'

    train_dataset, train_dataset_len = create_CitySegmentation(args, data_path=args.dataset, \
                        split=args.train_split, mode='train', transform=input_transform, \
                        base_size=args.base_size, crop_size=args.crop_size, batch_size=args.batch_size, \
                        device_num=args.group_size, rank=args.rank, shuffle=True)

    args.steps_per_epoch = math.ceil(train_dataset_len / args.batch_size / args.group_size)

    # create network
    f_model = FastSCNN(num_classes=19, aux=args.aux)

    # resume checkpoint if needed
    # resume checkpoint if needed
    if args.resume_path:
        if args.use_modelarts:
            import moxing as mox
            args.logger.info("copying resume checkpoint from obs to cache....")
            mox.file.copy_parallel(args.resume_path, 'cache/resume_path')
            args.logger.info("copying resume checkpoint finished....")
            args.resume_path = 'cache/resume_path/'

        args.resume_path = os.path.join(args.resume_path, args.resume_name)
        args.logger.info('loading resume checkpoint {} into network'.format(args.resume_path))
        load_param_into_net(f_model, load_checkpoint(args.resume_path))
        args.logger.info('loaded resume checkpoint {} into network'.format(args.resume_path))

    model = FastSCNNWithLossCell(f_model, args)
    model.set_train()

    # lr scheduling
    lr_list = LRScheduler(mode='cosine', base_lr=args.lr, nepochs=args.epochs, \
              iters_per_epoch=args.steps_per_epoch, power=0.9)(args.epochs*args.steps_per_epoch)

    # optimizer
    optimizer = nn.SGD(params=model.trainable_params(), momentum=args.momentum, \
                       learning_rate=Tensor(lr_list, mindspore.float32), \
                       weight_decay=args.weight_decay, loss_scale=1024)
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(model, optimizer=optimizer, loss_scale_manager=loss_scale, amp_level="O0")

    # define callbacks
    if args.rank == 0:
        time_cb = TimeMonitor(data_size=train_dataset_len)
        loss_cb = LossMonitor()
        callbacks = [time_cb, loss_cb]
    else:
        callbacks = None

    if args.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.steps_per_epoch*args.save_every,
                                       keep_checkpoint_max=args.ckpt_save_max)
        save_ckpt_path = os.path.join(save_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='rank_'+str(args.rank))
        callbacks.append(ckpt_cb)

    if args.eval_while_train == 1 and args.rank == 0:

        val_dataset, _ = create_CitySegmentation(args, data_path=args.dataset, \
                                       split='val', mode='val', transform=input_transform, \
                                       base_size=args.base_size, crop_size=args.crop_size, \
                                       batch_size=1, device_num=1, \
                                       rank=args.rank, shuffle=False)
        loss_f = TempLoss()
        network_eval = Model(f_model, loss_fn=loss_f, metrics={"SegmentationMetric": SegmentationMetric(19)})

        eval_param_dict = {"model": network_eval, "dataset": val_dataset}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=args.eval_steps,
                               eval_start_epoch=args.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=save_dir, besk_ckpt_name="best_map.ckpt",
                               metrics_name=("pixAcc", "mIou"))
        callbacks.append(eval_cb)

    model.train(args.epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=True)

    args.logger.info("training finished....")
    if args.use_modelarts:
        import moxing as mox
        args.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(save_dir, args.outer_path)
        args.logger.info("copying finished....")

if __name__ == '__main__':
    print('Starting training, Total Epochs: %d' % (args.epochs))
    train()
