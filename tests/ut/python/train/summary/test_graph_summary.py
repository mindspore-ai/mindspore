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
""" test_graph_summary """
import logging
import os
import numpy as np

import mindspore.nn as nn
from mindspore import Model, context
from mindspore.nn.optim import Momentum
from mindspore.train.summary import SummaryRecord
from mindspore.train.callback import SummaryCollector
from .....dataset_mock import MindData

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"
GRAPH_TEMP = CUR_DIR + "/ms_output-resnet50.pb"

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
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
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
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
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    return model


def get_dataset():
    """ get_datasetdataset """
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((2, 3, 224, 224), (2, 3))

    dataset = MindData(size=2, batch_size=2,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset


# Test 1: summary sample of graph
def test_graph_summary_sample():
    """ test_graph_summary_sample """
    log.debug("begin test_graph_summary_sample")
    dataset = get_dataset()
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), 0.1, 0.9)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_GRAPH", network=model._train_network) as test_writer:
        model.train(2, dataset)
        for i in range(1, 5):
            test_writer.record(i)


def test_graph_summary_callback():
    dataset = get_dataset()
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), 0.1, 0.9)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    summary_collector = SummaryCollector(SUMMARY_DIR,
                                         collect_freq=1,
                                         keep_default_action=False,
                                         collect_specified_data={'collect_graph': True})
    model.train(1, dataset, callbacks=[summary_collector])
