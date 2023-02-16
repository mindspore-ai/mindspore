# Copyright 2022 Huawei Technologies Co., Ltd
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

""" test OnRequestExit Callback"""

import os
import shutil
import signal
import sys
import time
from multiprocessing import Process
import numpy as np
import pytest

from mindspore import nn, context
from mindspore import dataset as ds
from mindspore.common.initializer import TruncatedNormal
from mindspore.train import Callback, OnRequestExit, LossMonitor, Model


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    LeNet5 network
    """

    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def generator_multi_column():
    i = 0
    while i < 1000:
        i += 1
        yield np.ones((1, 32, 32)).astype(np.float32) * 0.01, np.array(1).astype(np.int32)


def send_signal(sleep_time):
    time.sleep(sleep_time)
    os.kill(os.getppid(), signal.SIGTERM)


def construct_model():
    forward_net = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optim = nn.Momentum(forward_net.trainable_params(), 0.01, 0.9)
    model = Model(forward_net, loss_fn=loss, optimizer=optim, metrics={'accuracy'})
    return model


def construct_dataset():
    dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["data", "label"])
    dataset = dataset.batch(32, drop_remainder=True)
    return dataset


class EpochAndStepRecord(Callback):
    """Define EpochAndStepRecord Callback to record epoch and step"""

    def __init__(self):
        self.epoch = 0
        self.step = 0

    def on_train_end(self, run_context):
        cb_params = run_context.original_args()
        self.epoch = cb_params.cur_epoch_num
        self.step = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

    def on_eval_end(self, run_context):
        cb_params = run_context.original_args()
        self.step = cb_params.cur_step_num


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_on_request_exit_callback():
    """
    Feature: OnRequestExit Callback.
    Description: test OnRequestExit Callback when a signal receive.
    Expectation: When a signal received,
        the train process should be stopped and save the ckpt and mindir should be saved.
    """
    if sys.platform != 'linux':
        return
    context.set_context(mode=context.GRAPH_MODE)
    directory = "./data"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    dataset = construct_dataset()
    model = construct_model()
    loss_monitor = LossMonitor()
    epoch_and_step_record = EpochAndStepRecord()
    on_request_exit = OnRequestExit(save_ckpt=True, file_name='LeNet5', directory=directory)
    epoch_num = 100
    step_num = dataset.get_dataset_size()
    send_signal_process = Process(target=send_signal, args=[1])
    send_signal_process.start()
    model.train(epoch_num, dataset, callbacks=[loss_monitor, epoch_and_step_record, on_request_exit])
    train_ckpt_file = f"{directory}/LeNet5_train.ckpt"
    train_mindir_file = f"{directory}/LeNet5_train.mindir"
    eval_ckpt_file = f"{directory}/LeNet5_eval.ckpt"
    eval_mindir_file = f"{directory}/LeNet5_eval.mindir"
    assert epoch_and_step_record.epoch != epoch_num or epoch_and_step_record.step != step_num
    assert os.path.isfile(train_ckpt_file)
    assert os.path.isfile(train_mindir_file)
    ckpt_ctime = os.path.getctime(train_ckpt_file)
    mindir_ctime = os.path.getctime(train_mindir_file)

    dataset = construct_dataset()
    model = construct_model()
    on_request_exit = OnRequestExit(save_ckpt=True, file_name='LeNet5', directory=directory)
    send_signal_process = Process(target=send_signal, args=[0.5])
    send_signal_process.start()
    model.train(epoch_num, dataset, callbacks=[loss_monitor, epoch_and_step_record, on_request_exit])
    assert epoch_and_step_record.epoch != epoch_num or epoch_and_step_record.step != step_num
    assert os.path.getctime(train_ckpt_file) > ckpt_ctime
    assert os.path.getctime(train_mindir_file) > mindir_ctime

    send_signal_process = Process(target=send_signal, args=[0.14])
    send_signal_process.start()
    on_request_exit = OnRequestExit(save_ckpt=True, file_name='LeNet5', directory=directory)
    model.eval(dataset, callbacks=[loss_monitor, epoch_and_step_record, on_request_exit])
    assert epoch_and_step_record.step != step_num
    assert os.path.isfile(eval_ckpt_file)
    assert os.path.isfile(eval_mindir_file)

    dataset = construct_dataset()
    model = construct_model()
    on_request_exit = OnRequestExit(save_ckpt=True, file_name='LeNet5', directory=directory)
    model.eval(dataset, callbacks=[loss_monitor, epoch_and_step_record, on_request_exit])
    assert epoch_and_step_record.step == step_num

    shutil.rmtree(directory)
