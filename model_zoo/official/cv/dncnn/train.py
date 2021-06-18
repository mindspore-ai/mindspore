#!/usr/bin/env python3
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

import argparse
import datetime
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, LearningRateScheduler
from mindspore.train import Model
from mindspore.train.callback import Callback

from src.dataset import create_train_dataset
from src.model import DnCNN


class BatchAverageMSELoss(nn.Cell):
    def __init__(self, batch_size):
        super(BatchAverageMSELoss, self).__init__()
        self.batch_size = batch_size
        self.sumMSELoss = nn.MSELoss(reduction='sum')

    def construct(self, logits, labels):
        #equation 1 on the paper
        loss = self.sumMSELoss(logits, labels) / self.batch_size / 2
        return loss

class Print_info(Callback):
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print(datetime.datetime.now(), "end epoch", cb_params.cur_epoch_num)

def learning_rate_function(lr, cur_step_num):
    if cur_step_num % 40000 == 0:
        lr = lr*0.8
        print("current lr: ", str(lr))
    return lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DnCNN")
    parser.add_argument("--dataset_path", type=str, default="/code/BSR_bsds500/BSR/BSDS500/data/images/", \
                        help='training image path')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--model_type', type=str, default='DnCNN-S', \
                        choices=['DnCNN-S', 'DnCNN-B', 'DnCNN-3'], help='type of DnCNN')
    parser.add_argument('--noise_level', type=int, default=25, help="noise level only for DnCNN-S")
    parser.add_argument('--ckpt_prefix', type=str, default="dncnn_mindspore", help='ckpt name prefix')
    parser.add_argument('--epoch_num', type=int, default=50, help='epoch number')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    if args.model_type == 'DnCNN-S':
        network = DnCNN(1, num_of_layers=17)
    elif args.model_type == 'DnCNN-3' or args.model_type == 'DnCNN-B':
        network = DnCNN(1, num_of_layers=20)
    else:
        print("wrong model type")
        exit()

    ds_train = create_train_dataset(args.dataset_path, args.model_type, noise_level=args.noise_level, \
                                    batch_size=args.batch_size)
    opt = nn.AdamWeightDecay(network.trainable_params(), args.lr, weight_decay=args.weight_decay)
    loss_fun = BatchAverageMSELoss(args.batch_size)

    model = Model(network, loss_fun, opt)

    #training callbacks
    checkpoint_config = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=3)
    ckpoint_cb = ModelCheckpoint(prefix=args.ckpt_prefix, directory='./ckpt/', config=checkpoint_config)
    print_cb = Print_info()
    lr_cb = LearningRateScheduler(learning_rate_function)
    loss_monitor_cb = LossMonitor(per_print_times=100)

    print(datetime.datetime.now(), " training starts")
    model.train(args.epoch_num, ds_train, callbacks=[lr_cb, ckpoint_cb, print_cb, loss_monitor_cb], \
                dataset_sink_mode=False)
