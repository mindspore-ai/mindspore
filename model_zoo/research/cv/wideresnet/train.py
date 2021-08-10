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
#################train WideResNet example on cifar10########################
python train.py
"""
import ast
import os
import argparse
import numpy as np
from mindspore.common import set_seed
from mindspore import context
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.model import Model
import mindspore.nn as nn
import mindspore.common.initializer as weight_init

from src.wide_resnet import wideresnet
from src.dataset import create_dataset
from src.config import config_WideResnet as cfg
from src.generator_lr import get_lr
from src.cross_entropy_smooth import CrossEntropySmooth
from src.save_callback import SaveCallback

set_seed(1)

if __name__ == '__main__':

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    parser = argparse.ArgumentParser(description='Ascend WideResnet+CIFAR10 Training')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data')
    parser.add_argument('--ckpt_url', required=True, default=None, help='Location of ckpt.')
    parser.add_argument('--modelart', required=True, type=ast.literal_eval, default=False,
                        help='training on modelart or not, default is False')
    args = parser.parse_args()

    target = "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                        device_id=device_id)

    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    dataset_sink_mode = True

    if args.modelart:
        import moxing as mox
        data_path = '/cache/data_path'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=data_path)
    else:
        data_path = args.data_url

    ds_train = create_dataset(dataset_path=data_path,
                              do_train=True,
                              batch_size=cfg.batch_size)
    ds_eval = create_dataset(dataset_path=data_path,
                             do_train=False,
                             batch_size=cfg.batch_size)
    step_size = ds_train.get_dataset_size()

    net = wideresnet()

    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(gain=np.sqrt(2)),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=cfg.label_smooth_factor,
                              num_classes=cfg.num_classes)
    loss_scale = FixedLossScaleManager(loss_scale=cfg.loss_scale, drop_overflow_update=False)

    lr = get_lr(total_epochs=cfg.epoch_size, steps_per_epoch=step_size, lr_init=cfg.lr_init)
    lr = Tensor(lr)

    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': cfg.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params,
                   learning_rate=lr,
                   momentum=cfg.momentum,
                   loss_scale=cfg.loss_scale,
                   use_nesterov=True,
                   weight_decay=cfg.weight_decay)

    model = Model(net,
                  amp_level="O2",
                  loss_fn=loss,
                  optimizer=opt,
                  loss_scale_manager=loss_scale,
                  metrics={'accuracy'},
                  keep_batchnorm_fp32=False
                  )

    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    cb = [loss_cb, time_cb]
    ckpt_path = args.ckpt_url
    cb += [SaveCallback(model, ds_eval, ckpt_path, args.modelart)]

    model.train(epoch=cfg.epoch_size, train_dataset=ds_train, callbacks=cb,
                dataset_sink_mode=dataset_sink_mode)
