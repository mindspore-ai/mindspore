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
# ===========================================================================
"""DSCNN train."""
import os
import datetime
import argparse

import numpy as np
from mindspore import context
from mindspore import Tensor, Model
from mindspore.train.model import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size

from src.config import train_config
from src.log import get_logger
from src.dataset import audio_dataset
from src.ds_cnn import DSCNN
from src.loss import CrossEntropy
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.callback import ProgressMonitor, callback_func


def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def val(args, model, val_dataset):
    '''Eval.'''
    val_dataloader = val_dataset.create_tuple_iterator()
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    if args.amp_level == 'O0':
        origin_mstype = mstype.float32
    else:
        origin_mstype = mstype.float16
    model.predict_network.to_float(mstype.float32)

    for data, gt_classes in val_dataloader:
        output = model.predict(Tensor(data, mstype.float32))
        output = output.asnumpy()
        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[:, -5:]
        gt_classes = gt_classes.asnumpy()
        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += output.shape[0]

    model.predict_network.to_float(origin_mstype)
    results = [[top1_correct], [top5_correct], [img_tot]]

    results = np.array(results)

    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    acc5 = 100.0 * top5_correct / img_tot
    if acc1 > args.best_acc:
        args.best_acc = acc1
        args.best_epoch = args.epoch_cnt - 1
    args.logger.info('Eval: top1_cor:{}, top5_cor:{}, tot:{}, acc@1={:.2f}%, acc@5={:.2f}%' \
                     .format(top1_correct, top5_correct, img_tot, acc1, acc5))


def trainval(args, model, train_dataset, val_dataset, cb, rank):
    callbacks = callback_func(args, cb, 'epoch{}'.format(args.epoch_cnt))
    model.train(args.val_interval, train_dataset, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)
    if rank == 0:
        val(args, model, val_dataset)


def train():
    '''Train.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_distributed', action='store_true', default=False, help='distributed training')
    parser.add_argument('--device_id', type=int, default=0, help='which device the model will be trained on')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--amp_level', type=str, default='O0', choices=['O3', 'O2', 'O0'])
    args, model_settings = train_config(parser)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, enable_auto_mixed_precision=True)
    args.rank_save_ckpt_flag = 1

    # init distributed
    if args.is_distributed:
        if os.getenv('DEVICE_ID', "not_set").isdigit():
            context.set_context(device_id=int(os.getenv('DEVICE_ID')))
        init()
        rank = get_rank()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=group_size, gradients_mean=True)
    else:
        rank = 0
        group_size = 1
        context.set_context(device_id=args.device_id)
    # Logger
    args.outputs_dir = os.path.join(args.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir)

    # Dataloader: train, val
    train_dataset = audio_dataset(args.feat_dir, 'training', model_settings['spectrogram_length'],
                                  model_settings['dct_coefficient_count'], args.per_batch_size, group_size, rank)
    args.steps_per_epoch = train_dataset.get_dataset_size()
    val_dataset = audio_dataset(args.feat_dir, 'validation', model_settings['spectrogram_length'],
                                model_settings['dct_coefficient_count'], args.per_batch_size)

    # show args
    args.logger.save_args(args)

    # Network
    args.logger.important_info('start create network')
    network = DSCNN(model_settings, args.model_size_info)

    # Load pretrain model
    if os.path.isfile(args.pretrained):
        load_checkpoint(args.pretrained, network)
        args.logger.info('load model {} success'.format(args.pretrained))

    # Loss
    criterion = CrossEntropy(num_classes=model_settings['label_count'])

    # LR scheduler
    if args.lr_scheduler == 'multistep':
        lr_scheduler = MultiStepLR(args.lr, args.lr_epochs, args.lr_gamma, args.steps_per_epoch,
                                   args.max_epoch, warmup_epochs=args.warmup_epochs)
    elif args.lr_scheduler == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(args.lr, args.T_max, args.steps_per_epoch, args.max_epoch,
                                         warmup_epochs=args.warmup_epochs, eta_min=args.eta_min)
    else:
        raise NotImplementedError(args.lr_scheduler)
    lr_schedule = lr_scheduler.get_lr()

    # Optimizer
    opt = Momentum(params=network.trainable_params(),
                   learning_rate=Tensor(lr_schedule),
                   momentum=args.momentum,
                   weight_decay=args.weight_decay)

    model = Model(network, loss_fn=criterion, optimizer=opt, amp_level=args.amp_level, keep_batchnorm_fp32=False)

    # Training
    args.epoch_cnt = 0
    args.best_epoch = 0
    args.best_acc = 0
    progress_cb = ProgressMonitor(args)
    while args.epoch_cnt + args.val_interval < args.max_epoch:
        trainval(args, model, train_dataset, val_dataset, progress_cb, rank)
    rest_ep = args.max_epoch - args.epoch_cnt
    if rest_ep > 0:
        trainval(args, model, train_dataset, val_dataset, progress_cb, rank)

    args.logger.info('Best epoch:{} acc:{:.2f}%'.format(args.best_epoch, args.best_acc))

if __name__ == "__main__":
    train()
