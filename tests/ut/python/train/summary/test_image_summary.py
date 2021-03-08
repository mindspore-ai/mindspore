# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""test_image_summary"""
import logging
import os
import numpy as np

import mindspore.nn as nn
from mindspore import Model, context
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.train.summary.summary_record import SummaryRecord, _cache_summary_tensor_data
from mindspore.train.callback import Callback
from .....dataset_mock import MindData

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


def make_image_tensor(shape, dtype=float):
    """ make_image_tensor """
    number = np.prod(shape)
    x = (np.arange(number, dtype=dtype)).reshape(shape)
    return x


def get_test_data(step):
    """ get_test_data """
    test_data_list = []
    tag1 = "x1[:Image]"
    tag2 = "x2[:Image]"
    np1 = make_image_tensor([2, 3, 8, 8])
    np2 = make_image_tensor([step, 3, 8, 8])

    dict1 = {}
    dict1["name"] = tag1
    dict1["data"] = Tensor(np1)

    dict2 = {}
    dict2["name"] = tag2
    dict2["data"] = Tensor(np2)

    test_data_list.append(dict1)
    test_data_list.append(dict2)

    return test_data_list


# Test: call method on parse graph code
def test_image_summary_sample():
    """ test_image_summary_sample """
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_IMAGE") as test_writer:

        for i in range(1, 5):
            test_data = get_test_data(i)
            _cache_summary_tensor_data(test_data)
            test_writer.record(i)
            test_writer.flush()


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal',
                              pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


class LossNet(nn.Cell):
    """ LossNet definition """

    def __init__(self):
        super(LossNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal',
                              pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0
        self.loss = nn.SoftmaxCrossEntropyWithLogits()

    def construct(self, x, y):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.loss(x, y)
        return out


def get_model():
    """ get_model """
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    return model


def get_dataset():
    """ get_dataset """
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((2, 3, 224, 224), (2, 3))

    dataset = MindData(size=2, batch_size=2,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset


class ImageSummaryCallback(Callback):
    """Image summary callback."""

    def __init__(self, summary_record):
        self._summary_record = summary_record

    def __enter__(self):
        return self

    def __exit__(self, *err):
        self._summary_record.close()

    def record(self, step, train_network=None):
        """record data."""
        self._summary_record.record(step, train_network)
        self._summary_record.flush()


def test_image_summary_train():
    """ test_image_summary_train """
    dataset = get_dataset()
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_IMAGE") as test_writer:
        model = get_model()
        callback = ImageSummaryCallback(test_writer)
        model.train(2, dataset, callbacks=[callback])


def test_image_summary_data():
    """ test_image_summary_data """
    dataset = get_dataset()

    test_data_list = []
    i = 1
    for next_element in dataset:
        tag = "image_" + str(i) + "[:Image]"
        dct = {}
        dct["name"] = tag
        dct["data"] = Tensor(next_element[0])
        test_data_list.append(dct)
        i += 1

    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_IMAGE") as test_writer:
        _cache_summary_tensor_data(test_data_list)
        test_writer.record(1)
