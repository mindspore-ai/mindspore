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
"""train_imagenet."""
import os
import sys
import argparse
import random
import math
import numpy as np
from test_dataset import create_dataset
from config import config
from mindspore import context
from mindspore.nn.dynamic_lr import piecewise_constant_lr, warmup_lr
import mindspore.dataset.engine as de
from mindspore.train.serialization import load_checkpoint
from model.model import resnet50, TrainStepWrap, NetWithLossClass
from utils.distance import compute_dist, compute_score

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
args_opt = parser.parse_args()


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

local_data_url = 'data'
local_train_url = 'ckpt'


class Logger():
    '''Log'''
    def __init__(self, logFile="log_max.txt"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger("log/log.txt")


if __name__ == '__main__':
    query_dataset = create_dataset(data_dir=os.path.join('/home/dingfeifei/datasets', \
        'test/query'), p=config.p, k=config.k)
    gallery_dataset = create_dataset(data_dir=os.path.join('/home/dingfeifei/datasets', \
        'test/gallery'), p=config.p, k=config.k)

    epoch_size = config.epoch_size
    net = resnet50(class_num=config.class_num, is_train=False)
    loss_net = NetWithLossClass(net, is_train=False)

    base_lr = config.learning_rate
    warm_up_epochs = config.lr_warmup_epochs
    lr_decay_epochs = config.lr_decay_epochs
    lr_decay_factor = config.lr_decay_factor
    step_size = math.ceil(config.class_num / config.p)
    lr_decay_steps = []
    lr_decay = []
    for i, v in enumerate(lr_decay_epochs):
        lr_decay_steps.append(v * step_size)
        lr_decay.append(base_lr * lr_decay_factor ** i)
    lr_1 = warmup_lr(base_lr, step_size*warm_up_epochs, step_size, warm_up_epochs)
    lr_2 = piecewise_constant_lr(lr_decay_steps, lr_decay)
    lr = lr_1 + lr_2

    train_net = TrainStepWrap(loss_net, lr, config.momentum, is_train=False)

    load_checkpoint("checkpoints/40.ckpt", net=train_net)

    q_feats, q_labels, g_feats, g_labels = [], [], [], []
    for data, gt_classes, theta in query_dataset:
        output = train_net(data, gt_classes, theta)
        output = output.asnumpy()
        label = gt_classes.asnumpy()
        q_feats.append(output)
        q_labels.append(label)
    q_feats = np.vstack(q_feats)
    q_labels = np.hstack(q_labels)

    for data, gt_classes, theta in gallery_dataset:
        output = train_net(data, gt_classes, theta)
        output = output.asnumpy()
        label = gt_classes.asnumpy()
        g_feats.append(output)
        g_labels.append(label)
    g_feats = np.vstack(g_feats)
    g_labels = np.hstack(g_labels)

    q_g_dist = compute_dist(q_feats, g_feats, dis_type='cosine')
    mAP, cmc_scores = compute_score(q_g_dist, q_labels, g_labels)

    print(mAP, cmc_scores)
