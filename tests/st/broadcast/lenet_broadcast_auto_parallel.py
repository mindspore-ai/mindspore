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

import os

import numpy as np

import mindspore.communication.management as distributedTool
import mindspore.nn as nn
from mindspore import context
from mindspore.train import Accuracy
from mindspore.train import Model, LossMonitor, TimeMonitor
from tests.models.official.cv.lenet.src.dataset import create_dataset
from tests.models.official.cv.lenet.src.lenet import LeNet5

np.set_printoptions(threshold=np.inf)
device_num = 2
rank_id = 0


def setup_module():
    global device_num
    global rank_id
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    distributedTool.init()
    rank_id = distributedTool.get_rank()
    device_num = distributedTool.get_group_size()
    context.set_auto_parallel_context(device_num=device_num, parameter_broadcast=True)


def teardown_module():
    distributedTool.release()


def test_all_trains():
    ds_train = create_dataset(os.path.join('/home/workspace/mindspore_dataset/mnist', "train"), 32, 1)

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(1, ds_train, callbacks=[time_cb, LossMonitor()])
