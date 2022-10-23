# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Test export"""
import os
from io import BytesIO
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as CT
from mindspore.dataset.vision import Inter
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.train.serialization import export, _get_mindir_inputs, convert_model
from mindspore.nn import GraphCell


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


def create_dataset():
    # define dataset
    mnist_ds = ds.MnistDataset("../data/dataset/testMnistData")

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = CT.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label")
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image")

    # apply DatasetOps
    mnist_ds = mnist_ds.batch(batch_size=32, drop_remainder=True)

    return mnist_ds


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


class InputNet1(nn.Cell):
    def construct(self, x):
        return x


class InputNet2(nn.Cell):
    def construct(self, x, y):
        return x, y


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


def test_export_lenet_grad_mindir():
    """
    Feature: Export LeNet to MindIR
    Description: Test export API to save network into MindIR
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    network = LeNet5()
    network.set_train()
    predict = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([32, 10]).astype(np.float32))
    net = TrainOneStepCell(WithLossCell(network))
    file_name = "lenet_grad"
    export(net, predict, label, file_name=file_name, file_format='MINDIR')
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    os.remove(verify_name)


def test_get_mindir_inputs1():
    """
    Feature: Get MindIR input.
    Description: Test get mindir input.
    Expectation: Successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = InputNet1()
    input1 = Tensor(np.zeros([32, 10]).astype(np.float32))
    file_name = "input1.mindir"
    export(net, input1, file_name=file_name, file_format='MINDIR')
    input_tensor = _get_mindir_inputs(file_name)
    assert os.path.exists(file_name)
    assert input_tensor.shape == (32, 10)
    assert input_tensor.dtype == mindspore.float32
    os.remove(file_name)


def test_get_mindir_inputs2():
    """
    Feature: Get MindIR input.
    Description: Test get mindir input.
    Expectation: Successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = InputNet2()
    input1 = Tensor(np.zeros(1).astype(np.float16))
    input2 = Tensor(np.zeros([10, 20]), dtype=mstype.int32)
    file_name = "input2.mindir"
    export(net, input1, input2, file_name=file_name, file_format='MINDIR')
    input_tensor = _get_mindir_inputs(file_name)
    assert os.path.exists(file_name)
    assert len(input_tensor) == 2
    assert input_tensor[0].shape == (1,)
    assert input_tensor[0].dtype == mindspore.float16
    assert input_tensor[1].shape == (10, 20)
    assert input_tensor[1].dtype == mindspore.int32
    os.remove(file_name)


def test_convert_model():
    """
    Feature: Convert mindir to onnx.
    Description: Test convert.
    Expectation: Successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net1 = InputNet1()
    input1 = Tensor(np.ones([1, 32, 32]).astype(np.float32))
    mindir_name1 = "lenet1.mindir"
    export(net1, input1, file_name=mindir_name1, file_format='MINDIR')
    onnx_name1 = "lenet1.onnx"
    convert_model(mindir_name1, onnx_name1, "ONNX")
    assert os.path.exists(mindir_name1)
    assert os.path.exists(onnx_name1)
    os.remove(mindir_name1)
    os.remove(onnx_name1)

    net2 = InputNet2()
    input1 = Tensor(np.ones(32).astype(np.float32))
    input2 = Tensor(np.ones([32, 32]).astype(np.float32))
    mindir_name2 = "lenet2.mindir"
    export(net2, input1, input2, file_name=mindir_name2, file_format='MINDIR')
    onnx_name2 = "lenet2.onnx"
    convert_model(mindir_name2, onnx_name2, "ONNX")
    assert os.path.exists(mindir_name2)
    assert os.path.exists(onnx_name2)
    os.remove(mindir_name2)
    os.remove(onnx_name2)


def test_export_lenet_with_dataset():
    """
    Feature: Export LeNet with data preprocess to MindIR
    Description: Test export API to save network and dataset into MindIR
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    network = LeNet5()
    network.set_train()
    dataset = create_dataset()
    file_name = "lenet_preprocess"

    export(network, dataset, file_name=file_name, file_format='MINDIR')
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    os.remove(verify_name)


def test_export_lenet_onnx_with_encryption():
    """
    Feature: Export encrypted LeNet to ONNX
    Description: Test export API to save network with encryption into ONNX
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    network = LeNet5()
    network.set_train()
    file_name = "lenet_preprocess"

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    export(network, input_tensor, file_name=file_name, file_format='ONNX',
           enc_key=b'123456789', enc_mode=encrypt_func)
    verify_name = file_name + ".onnx"
    assert os.path.exists(verify_name)
    os.remove(verify_name)


def test_export_lenet_mindir_with_encryption():
    """
    Feature: Export encrypted LeNet to MindIR
    Description: Test export API to save network with encryption into MindIR
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    network = LeNet5()
    network.set_train()
    file_name = "lenet_preprocess"

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    export(network, input_tensor, file_name=file_name, file_format='MINDIR',
           enc_key=b'123456789', enc_mode=encrypt_func)
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    os.remove(verify_name)


def test_export_lenet_mindir_with_aes():
    """
    Feature: Export encrypted LeNet to MindIR
    Description: Test export API to save network with AES encryption into MindIR
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    network = LeNet5()
    file_name = "aes_encrypt"

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    export(network, input_tensor, file_name=file_name, file_format='MINDIR',
           enc_key=b'0123456789012345', enc_mode="AES-GCM")
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    load_graph = mindspore.load("aes_encrypt.mindir",
                                dec_key=b'0123456789012345', dec_mode="AES-GCM")
    load_net = nn.GraphCell(load_graph)
    os.remove(verify_name)

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    export(network, input_tensor, file_name=file_name, file_format='MINDIR',
           enc_key=b'0123456789012345', enc_mode="AES-CBC")
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    load_graph = mindspore.load("aes_encrypt.mindir",
                                dec_key=b'0123456789012345', dec_mode="AES-CBC")
    load_net = nn.GraphCell(load_graph)
    assert isinstance(load_net, GraphCell)
    os.remove(verify_name)


def test_export_lenet_mindir_with_sm4():
    """
    Feature: Export encrypted LeNet to MindIR
    Description: Test export API to save network with SM4-CBC encryption into MindIR
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    network = LeNet5()
    file_name = "sm4_encrypt"

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    export(network, input_tensor, file_name=file_name, file_format='MINDIR',
           enc_key=b'0123456789012345', enc_mode="SM4-CBC")
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    load_graph = mindspore.load("sm4_encrypt.mindir",
                                dec_key=b'0123456789012345', dec_mode="SM4-CBC")
    load_net = nn.GraphCell(load_graph)
    assert isinstance(load_net, GraphCell)
    os.remove(verify_name)
