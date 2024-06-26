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
import os
from io import BytesIO
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.train.serialization import export, load
from tests.mark_utils import arg_mark


def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class WithLossCell(nn.Cell):
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, label):
        predict = self.network(x)
        return self.loss(predict, label)


class TrainOneStepCell(nn.Cell):
    def __init__(self, network):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = nn.Momentum(self.weights, 0.1, 0.9)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True)

    def construct(self, x, label):
        weights = self.weights
        grads = self.grad(self.network, weights)(x, label)
        return self.optimizer(grads)


def encrypt_func(model_stream, key):
    plain_data = BytesIO()
    plain_data.write(model_stream)
    return plain_data.getvalue()


def decrypt_func(cipher_file, key):
    with open(cipher_file, 'rb') as f:
        plain_data = f.read()
    f.close()
    return plain_data


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_export_lenet_grad_mindir():
    """
    Feature: export LeNet to MindIR file
    Description: Test export API to export network into MindIR
    Expectation: export successfully
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = LeNet5()
    network.set_train()
    predict = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([32, 10]).astype(np.float32))
    net = TrainOneStepCell(WithLossCell(network))
    export(net, predict, label, file_name="lenet_grad", file_format='MINDIR')
    verify_name = "lenet_grad.mindir"
    assert os.path.exists(verify_name)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_load_mindir_and_run():
    """
    Feature: Load LeNet to MindIR
    Description: Test load API to load network into MindIR
    Expectation: load successfully
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = LeNet5()
    network.set_train()

    inputs0 = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    outputs0 = network(inputs0)

    inputs = Tensor(np.zeros([32, 1, 32, 32]).astype(np.float32))
    export(network, inputs, file_name="test_lenet_load", file_format='MINDIR')
    mindir_name = "test_lenet_load.mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(inputs0)
    assert np.allclose(outputs0.asnumpy(), outputs_after_load.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_load_mindir_and_run_with_encryption():
    """
    Feature: Load encrypted LeNet to MindIR with decryption
    Description: Test load API to load network with encryption into MindIR
    Expectation: load successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    network = LeNet5()
    network.set_train()

    inputs0 = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    outputs0 = network(inputs0)

    inputs = Tensor(np.zeros([32, 1, 32, 32]).astype(np.float32))
    export(network, inputs, file_name="test_lenet_load_enc", file_format='MINDIR',
           enc_key=b'123456789', enc_mode=encrypt_func)
    mindir_name = "test_lenet_load_enc.mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name, dec_key=b'123456789', dec_mode=decrypt_func)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(inputs0)
    assert np.allclose(outputs0.asnumpy(), outputs_after_load.asnumpy())
    os.remove(mindir_name)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_load_mindir_generated_from_old_version():
    """
    Feature: Load MindIR generated from old version
    Description: Test Load MindIR generated from old version
    Expectation: load successfully
    """
    context.set_context(mode=context.GRAPH_MODE)
    path = os.path.abspath(os.path.dirname(__file__)) + "/exported_mindir/old_version_mindir.mindir"
    graph = load(file_name=path)
    x = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    loaded_net = nn.GraphCell(graph)
    output = Tensor([0.01944945, 0.01933849, -0.00446877])
    output_from_load = loaded_net(x)
    assert np.allclose(output.asnumpy(), output_from_load.asnumpy())
