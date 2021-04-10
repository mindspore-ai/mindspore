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
import sys
import argparse
import random
import pickle
import numpy as np
from train_dataset import create_dataset
from config import config
from mindspore import context
from mindspore.nn.dynamic_lr import piecewise_constant_lr, warmup_lr
from mindspore.train.model import Model
from mindspore.train.serialization import load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor # TimeMonitor
import mindspore.dataset.engine as de
from mindspore.nn.metrics import Accuracy
from model.model import resnet50, NetWithLossClass, TrainStepWrap, TestStepWrap


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
    '''Logger'''
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
    epoch_size = config.epoch_size
    net = resnet50(class_num=config.class_num, is_train=True)
    loss_net = NetWithLossClass(net)

    dataset = create_dataset("/home/dingfeifei/datasets/faces_webface_112x112_raw_image", \
        p=config.p, k=config.k)

    step_size = dataset.get_dataset_size()
    base_lr = config.learning_rate
    warm_up_epochs = config.lr_warmup_epochs
    lr_decay_epochs = config.lr_decay_epochs
    lr_decay_factor = config.lr_decay_factor
    lr_decay_steps = []
    lr_decay = []
    for i, v in enumerate(lr_decay_epochs):
        lr_decay_steps.append(v * step_size)
        lr_decay.append(base_lr * lr_decay_factor ** i)
    lr_1 = warmup_lr(base_lr, step_size*warm_up_epochs, step_size, warm_up_epochs)
    lr_2 = piecewise_constant_lr(lr_decay_steps, lr_decay)
    lr = lr_1 + lr_2

    train_net = TrainStepWrap(loss_net, lr, config.momentum)
    test_net = TestStepWrap(net)

    f = open("checkpoints/pretrained_resnet50.pkl", "rb")
    param_dict = pickle.load(f)
    load_param_into_net(net=train_net, parameter_dict=param_dict)

    model = Model(train_net, eval_network=test_net, metrics={"Accuracy": Accuracy()})

    loss_cb = LossMonitor()
    cb = [loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps, \
        keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="resnet", directory='checkpoints/', \
        config=config_ck)
    cb += [ckpt_cb]
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
