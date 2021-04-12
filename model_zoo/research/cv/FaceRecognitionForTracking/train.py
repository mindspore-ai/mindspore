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
"""Face Recognition train."""
import os
import time
import argparse
import datetime
import warnings
import random
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, RunContext, _InternalCallbackParam, CheckpointConfig
from mindspore.nn.optim import SGD
from mindspore.nn import TrainOneStepCell
from mindspore.communication.management import get_group_size, init, get_rank

from src.dataset import get_de_dataset
from src.config import reid_1p_cfg, reid_8p_cfg
from src.lr_generator import step_lr
from src.log import get_logger, AverageMeter
from src.reid import SphereNet, CombineMarginFCFp16, BuildTrainNetworkWithHead
from src.loss import CrossEntropy

warnings.filterwarnings('ignore')
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True, device_id=devid)
random.seed(1)
np.random.seed(1)

def init_argument():
    """init config argument."""
    parser = argparse.ArgumentParser(description='Cifar10 classification')
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--data_dir', type=str, default='', help='image label list file, e.g. /home/label.txt')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')

    args = parser.parse_args()

    if args.is_distributed == 0:
        cfg = reid_1p_cfg
    else:
        cfg = reid_8p_cfg
    cfg.pretrained = args.pretrained
    cfg.data_dir = args.data_dir

    # Init distributed
    if args.is_distributed:
        init()
        cfg.local_rank = get_rank()
        cfg.world_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE

    # parallel_mode 'STAND_ALONE' do not support parameter_broadcast and mirror_mean
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=cfg.world_size,
                                      gradients_mean=True)

    mindspore.common.set_seed(1)

    # logger
    cfg.outputs_dir = os.path.join(cfg.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    cfg.logger = get_logger(cfg.outputs_dir, cfg.local_rank)

    # Show cfg
    cfg.logger.save_args(cfg)
    return cfg

def main():
    cfg = init_argument()
    loss_meter = AverageMeter('loss')
    # dataloader
    cfg.logger.info('start create dataloader')
    de_dataset, steps_per_epoch, class_num = get_de_dataset(cfg)
    cfg.steps_per_epoch = steps_per_epoch
    cfg.logger.info('step per epoch: %d' % cfg.steps_per_epoch)
    de_dataloader = de_dataset.create_tuple_iterator()
    cfg.logger.info('class num original: %d' % class_num)
    if class_num % 16 != 0:
        class_num = (class_num // 16 + 1) * 16
    cfg.class_num = class_num
    cfg.logger.info('change the class num to: %d' % cfg.class_num)
    cfg.logger.info('end create dataloader')

    # backbone and loss
    cfg.logger.important_info('start create network')
    create_network_start = time.time()

    network = SphereNet(num_layers=cfg.net_depth, feature_dim=cfg.embedding_size, shape=cfg.input_size)
    head = CombineMarginFCFp16(embbeding_size=cfg.embedding_size, classnum=cfg.class_num)
    criterion = CrossEntropy()

    # load the pretrained model
    if os.path.isfile(cfg.pretrained):
        param_dict = load_checkpoint(cfg.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        cfg.logger.info('load model %s success' % cfg.pretrained)

    # mixed precision training
    network.add_flags_recursive(fp16=True)
    head.add_flags_recursive(fp16=True)
    criterion.add_flags_recursive(fp32=True)

    train_net = BuildTrainNetworkWithHead(network, head, criterion)

    # optimizer and lr scheduler
    lr = step_lr(lr=cfg.lr, epoch_size=cfg.epoch_size, steps_per_epoch=cfg.steps_per_epoch, max_epoch=cfg.max_epoch,
                 gamma=cfg.lr_gamma)
    opt = SGD(params=train_net.trainable_params(), learning_rate=lr, momentum=cfg.momentum,
              weight_decay=cfg.weight_decay, loss_scale=cfg.loss_scale)

    # package training process, adjust lr + forward + backward + optimizer
    train_net = TrainOneStepCell(train_net, opt, sens=cfg.loss_scale)

    # checkpoint save
    if cfg.local_rank == 0:
        ckpt_max_num = cfg.max_epoch * cfg.steps_per_epoch // cfg.ckpt_interval
        train_config = CheckpointConfig(save_checkpoint_steps=cfg.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=train_config, directory=cfg.outputs_dir, prefix='{}'.format(cfg.local_rank))
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

    cfg.logger.important_info('====start train====')
    for i, total_data in enumerate(de_dataloader):
        data, gt = total_data
        data = Tensor(data)
        gt = Tensor(gt)

        loss = train_net(data, gt)
        loss_meter.update(loss.asnumpy())

        # ckpt
        if cfg.local_rank == 0:
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        # logging loss, fps, ...
        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            cfg.logger.important_info('{}, graph compile time={:.2f}s'.format(cfg.task, time_for_graph_compile))

        if i % cfg.log_interval == 0 and cfg.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / cfg.steps_per_epoch)
            fps = cfg.per_batch_size * (i - old_progress) * cfg.world_size / time_used
            cfg.logger.info('epoch[{}], iter[{}], {}, {:.2f} imgs/sec, lr={}'.format(epoch, i, loss_meter, fps, lr[i]))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if i % cfg.steps_per_epoch == 0 and cfg.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / cfg.steps_per_epoch)
            fps = cfg.per_batch_size * cfg.world_size * cfg.steps_per_epoch / epoch_time_used
            cfg.logger.info('=================================================')
            cfg.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            cfg.logger.info('=================================================')
            t_epoch = time.time()

    cfg.logger.important_info('====train end====')


if __name__ == "__main__":
    main()
