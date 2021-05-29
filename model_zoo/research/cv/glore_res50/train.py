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
################################train glore_resnet50################################
python train.py
"""
import os
import random
import argparse
import ast
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore import dataset as de
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init
import mindspore.common.initializer as weight_init
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.config import config
from src.lr_generator import get_lr
from src.dataset import create_train_dataset, create_eval_dataset, _get_rank_info
from src.save_callback import SaveCallback
from src.glore_resnet50 import glore_resnet50
from src.loss import SoftmaxCrossEntropyExpand, CrossEntropySmooth
from src.autoaugment import autoaugment

parser = argparse.ArgumentParser(description='Image classification with glore_resnet50')
parser.add_argument('--use_glore', type=ast.literal_eval, default=True, help='Enable GloreUnit')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
parser.add_argument('--data_url', type=str, default=None,
                    help='Dataset path')
parser.add_argument('--train_url', type=str)
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--is_modelarts', type=ast.literal_eval, default=True)
parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Pretrained ckpt path')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
args_opt = parser.parse_args()

if args_opt.is_modelarts:
    import moxing as mox

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
    else:
        if target == "Ascend":
            device_id = args_opt.device_id
            context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False, device_id=device_id)
    # create dataset
    train_dataset_path = os.path.join(args_opt.data_url, 'train')
    eval_dataset_path = os.path.join(args_opt.data_url, 'val')

    # download dataset from obs to cache if train on ModelArts
    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/train'
        eval_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/val'
    if config.use_autoaugment:
        print("===========Use autoaugment==========")
        train_dataset = autoaugment(dataset_path=train_dataset_path, repeat_num=1,
                                    batch_size=config.batch_size, target=target)
    else:
        train_dataset = create_train_dataset(dataset_path=train_dataset_path, repeat_num=1,
                                             batch_size=config.batch_size, target=target)

    eval_dataset = create_eval_dataset(dataset_path=eval_dataset_path, repeat_num=1, batch_size=config.batch_size)

    step_size = train_dataset.get_dataset_size()

    # define net

    net = glore_resnet50(class_num=config.class_num, use_glore=args_opt.use_glore)

    # init weight
    if config.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_ckpt)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv1d)):
                if config.weight_init == 'xavier_uniform':
                    cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(),
                                                                        cell.weight.shape,
                                                                        cell.weight.dtype)
                elif config.weight_init == 'he_uniform':
                    cell.weight.default_input = weight_init.initializer(weight_init.HeUniform(),
                                                                        cell.weight.shape,
                                                                        cell.weight.dtype)
                else:  # config.weight_init == 'he_normal' or the others
                    cell.weight.default_input = weight_init.initializer(weight_init.HeNormal(),
                                                                        cell.weight.shape,
                                                                        cell.weight.dtype)

            if isinstance(cell, nn.Dense):
                cell.weight.default_input = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)

    # init lr
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size, steps_per_epoch=step_size, lr_decay_mode=config.lr_decay_mode)
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
    net_opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    if config.use_label_smooth:
        loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyExpand(sparse=True)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale, metrics={"Accuracy": Accuracy()})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    device_num, device_id = _get_rank_info()
    if config.save_checkpoint:
        if args_opt.is_modelarts:
            save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
        else:
            save_checkpoint_path = config.save_checkpoint_path
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="glore_resnet50", directory=save_checkpoint_path, config=config_ck)
        save_cb = SaveCallback(model, eval_dataset, save_checkpoint_path)
        cb += [ckpt_cb, save_cb]

    # train model
    print("=======Training Begin========")
    model.train(config.epoch_size - config.pretrain_epoch_size, train_dataset, callbacks=cb, dataset_sink_mode=True)

    # copy train result from cache to obs
    mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)
    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)
