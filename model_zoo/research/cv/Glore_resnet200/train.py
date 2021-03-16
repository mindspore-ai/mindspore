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
#################train glore_resnet200 on Imagenet2012########################
python train.py
"""
import os
import random
import argparse
import ast
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as de
from mindspore.train.model import Model, ParallelMode
from mindspore.communication import management as MultiAscend
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr
from src.glore_resnet200 import glore_resnet200
from src.dataset import create_dataset_ImageNet as get_dataset
from src.config import config
from src.loss import SoftmaxCrossEntropyExpand


parser = argparse.ArgumentParser(description='Image classification with glore_resnet200')
parser.add_argument('--use_glore', type=ast.literal_eval, default=True, help='Enable GloreUnit')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
parser.add_argument('--data_url', type=str, default=None,
                    help='Dataset path')
parser.add_argument('--train_url', type=str)
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--pre_trained', type=ast.literal_eval, default=False)
parser.add_argument('--pre_ckpt_path', type=str,
                    default='')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=True)
args_opt = parser.parse_args()

if args_opt.isModelArts:
    import moxing as mox

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if __name__ == '__main__':

    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
    else:
        if target == "Ascend":
            device_id = args_opt.device_id
            context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                                device_id=device_id)

    train_dataset_path = args_opt.data_url
    if args_opt.isModelArts:
        # download dataset from obs to cache
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID')
    # create dataset
    dataset = get_dataset(dataset_path=train_dataset_path, do_train=True, use_randaugment=True, repeat_num=1,
                          batch_size=config.batch_size, target=target)
    step_size = dataset.get_dataset_size()

    # define net

    net = glore_resnet200(class_num=config.class_num, use_glore=args_opt.use_glore)

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_ckpt_path)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)
            if isinstance(cell, nn.Dense):
                cell.weight.default_input = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    #
    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    net_opt = nn.SGD(group_params, learning_rate=lr, momentum=config.momentum, weight_decay=config.weight_decay,
                     loss_scale=config.loss_scale, nesterov=True)

    # define loss, model
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    rank_size = os.getenv("RANK_SIZE")
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        if args_opt.isModelArts:
            save_checkpoint_path = '/cache/train_output/checkpoint'
            if rank_size is None or int(rank_size) == 1:
                ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if rank_size is not None and int(rank_size) > 1 and MultiAscend.get_rank() % 8 == 0:
                ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
        else:
            if rank_size is None or int(rank_size) == 1:
                ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                          directory=os.path.join('./', 'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                          config=config_ck)
                cb += [ckpt_cb]
            if rank_size is not None and int(rank_size) > 1 and MultiAscend.get_rank() % 8 == 0:
                ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                          directory=os.path.join('./', 'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                          config=config_ck)
                cb += [ckpt_cb]

    model.train(config.epoch_size - config.pretrain_epoch_size, dataset,
                callbacks=cb, dataset_sink_mode=True)
    if args_opt.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output/checkpoint', dst_url=args_opt.train_url)
