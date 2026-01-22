# Copyright 2025 Huawei Technologies Co., Ltd
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
# limitations under the License

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, Momentum, WithLossCell
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Lamb
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.nn.utils import no_init_parameters
from tests.dataset_mock import MindData
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files, \
    find_ir_file_path, check_node_attrs_pair


def setup_function():
    keyword = 'fusion'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'fusion'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(128, 768, activation='relu')
        self.fc2 = nn.Dense(128, 768, activation='relu')
        self.fc3 = nn.Dense(128, 768, activation='relu')
        self.fc4 = nn.Dense(768, 768, activation='relu')
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class DenseNet1(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet1, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc2 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc3 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc4 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(q)
        v = self.fc3(k)
        s = self.fc4(v)
        return s


class DenseNet2(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet2, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc2 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc3 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc4 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc5 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc6 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc7 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc8 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(q)
        v = self.fc3(k)
        s = self.fc4(v)
        t = self.fc5(s)
        u = self.fc6(t)
        w = self.fc7(u)
        z = self.fc8(w)
        return z


class SimpleDMLNet(nn.Cell):
    def __init__(self, net1, net2):
        super(SimpleDMLNet, self).__init__()
        self.backbone1 = net1
        self.backbone2 = net2

    def construct(self, x):
        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        return x1 + x2


def train_common(net, parallel_config):
    init_hccl(global_rank=0, device_num=4)

    batch_size = 32
    predict = Tensor(np.ones([batch_size, 128]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    with no_init_parameters():
        opt = Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    # set auto_parallel config
    net = set_parallel_mode(net, parallel_config)
    model = Model(net, loss, opt)
    model.train(epoch=2, train_dataset=dataset, dataset_sink_mode=False)

    allreduce_fusion_dict = _cell_graph_executor._get_allreduce_fusion(model._train_network)
    print(f"allreduce_fusion_dict is {allreduce_fusion_dict}")
    return allreduce_fusion_dict


def test_allreduce_fusion_auto():
    """
    Feature: test_allreduce_fusion in auto mode
    Description: allreduce fusion in auto mode
    Expectation: success
    """
    graph_path = './test_auto_parallel/test_allreduce_fusion_auto_graphs'
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    with no_init_parameters():
        net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    comm_fusion_dict = {"allreduce": {"mode": "auto", "config": None}}
    parallel_config = {"parallel_mode": "semi_auto", "comm_fusion": comm_fusion_dict}
    train_common(net, parallel_config)
    validate_ir = find_ir_file_path(graph_path, "validate")
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    check_pairs = {"AllReduce": expect_dict}
    check_node_attrs_pair(validate_ir, check_pairs)


def test_allreduce_fusion_size():
    """
    Feature: test_allreduce_fusion in size mode
    Description: allreduce fusion in size mode
    Expectation: success
    """
    graph_path = './test_auto_parallel/test_allreduce_fusion_size_graphs'
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    comm_fusion_dict = {"allreduce": {"mode": "size", "config": 32}}
    parallel_config = {"parallel_mode": "semi_auto", "comm_fusion": comm_fusion_dict}

    with no_init_parameters():
        net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    train_common(net, parallel_config)
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    validate_ir = find_ir_file_path(graph_path, "validate")
    check_pairs = {"AllReduce": expect_dict}
    check_node_attrs_pair(validate_ir, check_pairs)


def test_lamb_split_fusion_in_index():
    """
    Feature: test_allreduce_fusion in index mode
    Description: allreduce fusion in index mode
    Expectation: success
    """
    graph_path = './test_auto_parallel/test_lamb_split_fusion_in_index_graphs'
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    with no_init_parameters():
        net = Net()
        optimizer = Lamb(net.trainable_params(), learning_rate=0.1)

    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)

    # set auto_parallel
    init_hccl(global_rank=0, device_num=2)
    comm_fusion_dict = {"allreduce": {"mode": "index", "config": [2, 4, 6, 8]}}
    parallel_config = {"parallel_mode": "semi_auto", "enable_parallel_optimizer": True, "comm_fusion": comm_fusion_dict,
                       "dataset_strategy": "data_parallel"}
    train_network = set_parallel_mode(train_network, parallel_config)

    # compile
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    _cell_graph_executor.compile(train_network, inputs, label)

    # validation
    validate_ir = find_ir_file_path(graph_path, "validate")
    check_pairs = {"AllReduce": {"fusion: I64(1)": 4}}
    check_node_attrs_pair(validate_ir, check_pairs)
