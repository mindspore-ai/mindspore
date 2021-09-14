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
import numpy as np
from mindspore import context
from mindspore import Tensor, Model
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint
from mindspore.communication.management import init
from src.log import get_logger
from src.dataset import audio_dataset
from src.ds_cnn import DSCNN
from src.loss import CrossEntropy
from src.lr_scheduler import MultiStepLR, CosineAnnealingLR
from src.callback import ProgressMonitor, callback_func
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num


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


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    '''Train.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    config.rank_save_ckpt_flag = 1

    # init distributed
    if config.is_distributed:
        if get_device_id():
            context.set_context(device_id=get_device_id())
        init()
        rank = get_rank_id()
        device_num = get_device_num()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=device_num, gradients_mean=True)
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=get_device_id())
    # Logger
    config.outputs_dir = os.path.join(config.save_ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir)

    # Dataloader: train, val
    train_dataset = audio_dataset(config.train_feat_dir, 'training', config.model_setting_spectrogram_length,
                                  config.model_setting_dct_coefficient_count, config.per_batch_size, device_num, rank)
    config.steps_per_epoch = train_dataset.get_dataset_size()
    val_dataset = audio_dataset(config.train_feat_dir, 'validation', config.model_setting_spectrogram_length,
                                config.model_setting_dct_coefficient_count, config.per_batch_size)

    # show args
    config.logger.save_args(config)

    # Network
    config.logger.important_info('start create network')
    network = DSCNN(config, config.model_size_info)

    # Load pretrain model
    if os.path.isfile(config.pretrained):
        load_checkpoint(config.pretrained, network)
        config.logger.info('load model %s success', config.pretrained)

    # Loss
    criterion = CrossEntropy(num_classes=config.model_setting_label_count)

    # LR scheduler
    if config.lr_scheduler == 'multistep':
        lr_scheduler = MultiStepLR(config.lr, config.lr_epochs, config.lr_gamma, config.steps_per_epoch,
                                   config.max_epoch, warmup_epochs=config.warmup_epochs)
    elif config.lr_scheduler == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(config.lr, config.T_max, config.steps_per_epoch, config.max_epoch,
                                         warmup_epochs=config.warmup_epochs, eta_min=config.eta_min)
    else:
        raise NotImplementedError(config.lr_scheduler)
    lr_schedule = lr_scheduler.get_lr()

    # Optimizer
    opt = Momentum(params=network.trainable_params(),
                   learning_rate=Tensor(lr_schedule),
                   momentum=config.momentum,
                   weight_decay=config.weight_decay)

    model = Model(network, loss_fn=criterion, optimizer=opt, amp_level=config.amp_level, keep_batchnorm_fp32=False)

    # Training
    config.epoch_cnt = 0
    config.best_epoch = 0
    config.best_acc = 0
    progress_cb = ProgressMonitor(config)
    while config.epoch_cnt + config.val_interval < config.max_epoch:
        trainval(config, model, train_dataset, val_dataset, progress_cb, rank)
    rest_ep = config.max_epoch - config.epoch_cnt
    if rest_ep > 0:
        trainval(config, model, train_dataset, val_dataset, progress_cb, rank)

    config.logger.info('Best epoch:{} acc:{:.2f}%'.format(config.best_epoch, config.best_acc))


if __name__ == "__main__":
    train()
