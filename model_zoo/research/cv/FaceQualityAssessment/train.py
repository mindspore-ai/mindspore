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
"""Face Quality Assessment train."""
import os
import time
import datetime
import argparse
import warnings
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, RunContext, _InternalCallbackParam, CheckpointConfig
from mindspore.nn import TrainOneStepCell
from mindspore.nn.optim import Momentum
from mindspore.communication.management import get_group_size, init, get_rank

from src.loss import CriterionsFaceQA
from src.config import faceqa_1p_cfg, faceqa_8p_cfg
from src.face_qa import FaceQABackbone, BuildTrainNetwork
from src.lr_generator import warmup_step
from src.dataset import faceqa_dataset
from src.log import get_logger, AverageMeter

warnings.filterwarnings('ignore')
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)
mindspore.common.seed.set_seed(1)

def main(args):

    if args.is_distributed == 0:
        cfg = faceqa_1p_cfg
    else:
        cfg = faceqa_8p_cfg

    cfg.data_lst = args.train_label_file
    cfg.pretrained = args.pretrained

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
    loss_meter = AverageMeter('loss')

    # Dataloader
    cfg.logger.info('start create dataloader')
    de_dataset = faceqa_dataset(imlist=cfg.data_lst, local_rank=cfg.local_rank, world_size=cfg.world_size,
                                per_batch_size=cfg.per_batch_size)
    cfg.steps_per_epoch = de_dataset.get_dataset_size()
    de_dataset = de_dataset.repeat(cfg.max_epoch)
    de_dataloader = de_dataset.create_tuple_iterator(output_numpy=True)
    # Show cfg
    cfg.logger.save_args(cfg)
    cfg.logger.info('end create dataloader')

    # backbone and loss
    cfg.logger.important_info('start create network')
    create_network_start = time.time()

    network = FaceQABackbone()
    criterion = CriterionsFaceQA()

    # load pretrain model
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
        cfg.logger.info('load model %s success.' % cfg.pretrained)

    # optimizer and lr scheduler
    lr = warmup_step(cfg, gamma=0.9)
    opt = Momentum(params=network.trainable_params(),
                   learning_rate=lr,
                   momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay,
                   loss_scale=cfg.loss_scale)

    # package training process, adjust lr + forward + backward + optimizer
    train_net = BuildTrainNetwork(network, criterion)
    train_net = TrainOneStepCell(train_net, opt, sens=cfg.loss_scale,)

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
    for i, (data, gt) in enumerate(de_dataloader):
        # clean grad + adjust lr + put data into device + forward + backward + optimizer, return loss
        data = data.astype(np.float32)
        gt = gt.astype(np.float32)
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
            cfg.logger.info('epoch[{}], iter[{}], {}, {:.2f} imgs/sec'.format(epoch, i, loss_meter, fps))
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
    parser = argparse.ArgumentParser(description='Face Quality Assessment')
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--train_label_file', type=str, default='', help='image label list file, e.g. /home/label.txt')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')

    arg = parser.parse_args()

    main(arg)
