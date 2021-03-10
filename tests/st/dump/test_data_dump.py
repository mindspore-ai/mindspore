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
import json
import time
import shutil
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.nn import Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)

x = np.random.randn(1, 3, 3, 4).astype(np.float32)
y = np.random.randn(1, 3, 3, 4).astype(np.float32)

def change_current_dump_json(file_name, dump_path):
    with open(file_name, 'r+') as f:
        data = json.load(f)

    data["common_dump_settings"]["path"] = dump_path
    with open(file_name, 'w') as f:
        json.dump(data, f)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_async_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    pwd = os.getcwd()
    dump_path = pwd + "/async_dump"
    change_current_dump_json('async_dump.json', dump_path)
    os.environ['MINDSPORE_DUMP_CONFIG'] = pwd + "/async_dump.json"
    device_id = context.get_context("device_id")
    dump_file_path = pwd + '/async_dump/device_{}/Net_graph_0/0/0/'.format(device_id)
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)
    add = Net()
    add(Tensor(x), Tensor(y))
    time.sleep(5)
    assert len(os.listdir(dump_file_path)) == 1

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_e2e_dump():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    pwd = os.getcwd()
    dump_path = pwd + "/e2e_dump"
    change_current_dump_json('e2e_dump.json', dump_path)
    os.environ['MINDSPORE_DUMP_CONFIG'] = pwd + "/e2e_dump.json"
    device_id = context.get_context("device_id")
    dump_file_path = pwd + '/e2e_dump/Net/device_{}/iteration_1/'.format(device_id)
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)
    add = Net()
    add(Tensor(x), Tensor(y))
    time.sleep(5)
    assert len(os.listdir(dump_file_path)) == 5

class ReluReduceMeanDenseRelu(Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.dense = Dense(in_channel, num_class, kernel, bias)

    def construct(self, x_):
        x_ = self.relu(x_)
        x_ = self.mean(x_, (2, 3))
        x_ = self.dense(x_)
        x_ = self.relu(x_)
        return x_


def search_path(path, keyword):
    content = os.listdir(path)
    for each in content:
        each_path = path + os.sep + each
        if keyword in each:
            return each_path
        read_write = os.access(each_path, os.W_OK) and os.access(each_path, os.R_OK)
        if not read_write:
            continue
        if os.path.isdir(each_path):
            search_path(each_path, keyword)
    return None

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_async_dump_net_multi_layer_mode1():
    test_name = "test_async_dump_net_multi_layer_mode1"
    json_file = os.path.join(os.getcwd(), "{}.json".format(test_name))
    device_id = context.get_context("device_id")
    dump_full_path = os.path.join("/tmp/async_dump/", "{}_{}".format(test_name, device_id))
    os.system("rm -rf {}/*".format(dump_full_path))
    os.environ["MINDSPORE_DUMP_CONFIG"] = json_file
    weight = Tensor(np.ones((1000, 2048)).astype(np.float32))
    bias = Tensor(np.ones((1000,)).astype(np.float32))
    net = ReluReduceMeanDenseRelu(weight, bias, 2048, 1000)
    criterion = SoftmaxCrossEntropyWithLogits(sparse=False)
    optimizer = Momentum(learning_rate=0.1, momentum=0.1, params=filter(lambda  x: x.requires_grad, net.get_parameters()))
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    inputs = Tensor(np.random.randn(32, 2048, 7, 7).astype(np.float32))
    label = Tensor(np.zeros(shape=(32, 1000)).astype(np.float32))
    net_dict = train_network(inputs, label)

    dump_path = "/tmp/async_dump/{}/device_{}/test_graph_0/0/0/".format(test_name, device_id)
    dump_file = os.listdir(dump_path)
    dump_file_name = ""
    for file in dump_file:
        if "SoftmaxCrossEntropyWithLogits" in file:
            dump_file_name = file
    dump_file_full_path = os.path.join(dump_path, dump_file_name)
    npy_path = os.path.join(os.getcwd(), "./{}".format(test_name))
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
    os.mkdir(npy_path)
    tool_path = search_path('/usr/local/Ascend', 'msaccucmp.pyc')
    if tool_path:
        cmd = "python {0} convert -d {1} -out {2}".format(tool_path, dump_file_full_path, npy_path)
        os.system(cmd)
        npy_file_list = os.listdir(npy_path)
        dump_result = {}
        for file in npy_file_list:
            if "output.0.npy" in file:
                dump_result["output0"] = np.load(os.path.join(npy_path, file))
        for index, value in enumerate(net_dict):
            assert value.asnumpy() == dump_result["output0"][index]
    else:
        print('not find convert tools msaccucmp.pyc')
