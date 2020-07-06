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
"""Warpctc training"""
import os
import math as m
import random
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore import dataset as de
from mindspore.train.model import Model, ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init

from src.loss import CTCLoss
from src.config import config as cf
from src.dataset import create_dataset
from src.warpctc import StackedRNN
from src.warpctc_for_train import TrainOneStepCellWithGradClip
from src.lr_schedule import get_lr

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description="Warpctc training")
parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default is false.")
parser.add_argument('--device_num', type=int, default=1, help='Device num, default is 1.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path, default is None')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    save_graphs=False,
                    device_id=device_id)

if __name__ == '__main__':
    if args_opt.run_distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=args_opt.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True)
        init()
    max_captcha_digits = cf.max_captcha_digits
    input_size = m.ceil(cf.captcha_height / 64) * 64 * 3
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, repeat_num=cf.epoch_size, batch_size=cf.batch_size)
    step_size = dataset.get_dataset_size()
    # define lr
    lr_init = cf.learning_rate if not args_opt.run_distribute else cf.learning_rate * args_opt.device_num
    lr = get_lr(cf.epoch_size, step_size, lr_init)
    # define loss
    loss = CTCLoss(max_sequence_length=cf.captcha_width, max_label_length=max_captcha_digits, batch_size=cf.batch_size)
    # define net
    net = StackedRNN(input_size=input_size, batch_size=cf.batch_size, hidden_size=cf.hidden_size)
    # define opt
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=cf.momentum)
    net = WithLossCell(net, loss)
    net = TrainOneStepCellWithGradClip(net, opt).set_train()
    # define model
    model = Model(net)
    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if cf.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=cf.save_checkpoint_steps,
                                     keep_checkpoint_max=cf.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="waptctc", directory=cf.save_checkpoint_path, config=config_ck)
        callbacks.append(ckpt_cb)
    model.train(cf.epoch_size, dataset, callbacks=callbacks)
