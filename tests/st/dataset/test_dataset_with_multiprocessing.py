# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import time
import subprocess

import cv2
import numpy as np
import psutil
import pytest

import mindspore as ms
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.train.callback import Callback, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.metrics import Accuracy
from tests.mark_utils import arg_mark


ms.set_seed(1)


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 64

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(256, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


class EvalCall(Callback):
    def __init__(self, model, dataset_val, data_size):
        super(EvalCall, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.data_size = data_size
        self.count = 0
        self.step = 0
        self.fds = 0
        self.lsof = 0

    def step_end(self, run_context):
        self.step += 1
        if self.step % self.data_size == 0:
            print('Begin eval ...')
            self.model.eval(self.dataset_val)

            time.sleep(1)
            if self.count == 0:
                self.fds = psutil.Process(os.getpid()).num_fds()
                self.lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
                print("eval: {}, file descriptor: {}, lsof files: {}".format(self.count, self.fds, self.lsof))
                self.count += 1
            else:
                fds = psutil.Process(os.getpid()).num_fds()
                lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
                print("eval: {}, file descriptor: {}, lsof files: {}".format(self.count, fds, lsof))
                assert self.fds == fds
                assert self.lsof == lsof
                self.count += 1


class Config:
    def __init__(self):
        self.device_num = 1
        self.device_target = "Ascend"
        self.all_reduce_fusion_config = [85, 160]
        self.batch_size = 2
        self.train_image_size = 28
        self.run_distribute = True
        self.class_num = 1001
        self.lr_init = 0.04
        self.lr_end_kf = 0.0
        self.lr_max_kf = 0.4
        self.lr_end_ft = 0.0
        self.lr_max_ft = 0.8
        self.epoch_kf = 2
        self.epoch_ft = 1
        self.momentum = 0.9
        self.loss_scale = 1024
        self.keep_checkpoint_max = 10


def create_dataset():
    # Iterable object as input source
    class Iterable:
        def __init__(self):
            self._data = np.ones((640, 50, 50, 3), dtype=np.uint8)
            self._label = np.ones((640,), dtype=np.int32)

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    dataset = ds.GeneratorDataset(Iterable(), column_names=["data", "label"], num_parallel_workers=4)

    def transform(data, label):
        data = cv2.resize(data, (28, 28))
        return data, label

    dataset = dataset.map(operations=transform, input_columns=["data", "label"], num_parallel_workers=4,
                          python_multiprocessing=True)

    rescale = 1.0 / 255.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081
    rescale_nml_op = vision.Rescale(rescale_nml * rescale, shift_nml)
    type_cast_op = transforms.TypeCast(mstype.int32)
    hwc2chw_op = vision.HWC2CHW()
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    dataset = dataset.map(operations=rescale_nml_op, input_columns="data")
    dataset = dataset.map(operations=hwc2chw_op, input_columns="data")
    dataset = dataset.batch(64, drop_remainder=True)
    return dataset


def set_parameter(config):
    """set_parameter"""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=config.device_target, save_graphs=False)


@pytest.mark.skip(reason="to be adjust case")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_network_dataset_with_multiprocessing_without_fd_leak():
    """
    Feature: Network with dataset which use multiprocessing to process data
    Description: Run eval in callback will create multi dataset iterator
    Expectation: File descriptors are not leaked
    """
    config = Config()
    set_parameter(config)
    train_dataset = create_dataset()
    eval_dataset = create_dataset()
    net = LeNet()

    # apply golden-stick algo
    lr = 0.001

    optimizer = nn.Momentum(filter(lambda p: p.requires_grad, net.get_parameters()),
                            learning_rate=lr,
                            momentum=config.momentum,
                            loss_scale=config.loss_scale
                            )

    kf_loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_cb = LossMonitor()
    cb = [loss_cb]

    model = ms.Model(net, loss_fn=kf_loss_fn, optimizer=optimizer, metrics={"Accuracy": Accuracy()})

    eval_cb = EvalCall(model, eval_dataset, 5)
    cb += [eval_cb]

    model.train(3, train_dataset, callbacks=cb, dataset_sink_mode=True)


@pytest.mark.skip(reason="to be adjust case")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_only_dataset_with_multiprocessing_without_fd_leak():
    """
    Feature: Only create dataset with iterator
    Description: Testing file handle management under multi-process
    Expectation: File descriptors are not leaked
    """
    train_dataset = create_dataset()

    init_fds = 0
    lsof = 0
    for epoch in range(5):
        for _ in train_dataset.create_tuple_iterator():
            pass

        time.sleep(1)
        if epoch == 0:
            init_fds = psutil.Process(os.getpid()).num_fds()
            init_lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
            print("epoch: {}, file descriptor: {}, lsof files: {}".format(epoch, init_fds, init_lsof), flush=True)
        else:
            fds = psutil.Process(os.getpid()).num_fds()
            lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
            print("epoch: {}, file descriptor: {}, lsof files: {}".format(epoch, fds, lsof), flush=True)
            assert init_fds == fds
            assert init_lsof == lsof


if __name__ == '__main__':
    test_network_dataset_with_multiprocessing_without_fd_leak()
    test_only_dataset_with_multiprocessing_without_fd_leak()
