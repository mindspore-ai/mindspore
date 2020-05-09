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
"""ut for model serialize(save/load)"""
import os
import stat
import time

import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.train.callback import _CheckpointManager
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net, \
    _exec_save_checkpoint, export, _save_graph
from ..ut_filter import run_on_onnxruntime, non_graph_engine

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    """Net definition."""

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, weight_init="zeros")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(int(224 * 224 * 64 / 16), num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


_input_x = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
_cur_dir = os.path.dirname(os.path.realpath(__file__))


def setup_module():
    import shutil
    if os.path.exists('./test_files'):
        shutil.rmtree('./test_files')


def test_save_graph():
    """ test_exec_save_graph """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.add = P.TensorAdd()

        def construct(self, x, y):
            z = self.add(x, y)
            return z

    net = Net()
    net.set_train()
    out_me_list = []
    x = Tensor(np.random.rand(2, 1, 2, 3).astype(np.float32))
    y = Tensor(np.array([1.2]).astype(np.float32))
    out_put = net(x, y)
    _save_graph(network=net, file_name="net-graph.meta")
    out_me_list.append(out_put)


def test_save_checkpoint():
    """ test_save_checkpoint """
    parameter_list = []
    one_param = {}
    param1 = {}
    param2 = {}
    one_param['name'] = "param_test"
    one_param['data'] = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), dtype=mstype.float32)
    param1['name'] = "param"
    param1['data'] = Tensor(np.random.randint(0, 255, [12, 1024]), dtype=mstype.float32)
    param2['name'] = "new_param"
    param2['data'] = Tensor(np.random.randint(0, 255, [12, 1024, 1]), dtype=mstype.float32)
    parameter_list.append(one_param)
    parameter_list.append(param1)
    parameter_list.append(param2)

    if os.path.exists('./parameters.ckpt'):
        os.chmod('./parameters.ckpt', stat.S_IWRITE)
        os.remove('./parameters.ckpt')

    ckpoint_file_name = os.path.join(_cur_dir, './parameters.ckpt')
    save_checkpoint(parameter_list, ckpoint_file_name)


def test_load_checkpoint_error_filename():
    ckpoint_file_name = 1
    with pytest.raises(ValueError):
        load_checkpoint(ckpoint_file_name)


def test_load_checkpoint():
    ckpoint_file_name = os.path.join(_cur_dir, './parameters.ckpt')
    par_dict = load_checkpoint(ckpoint_file_name)

    assert len(par_dict) == 3
    assert par_dict['param_test'].name == 'param_test'
    assert par_dict['param_test'].data.dtype() == mstype.float32
    assert par_dict['param_test'].data.shape() == (1, 3, 224, 224)
    assert isinstance(par_dict, dict)


def test_checkpoint_manager():
    """ test_checkpoint_manager """
    ckp_mgr = _CheckpointManager()

    ckpoint_file_name = os.path.join(_cur_dir, './test1.ckpt')
    with open(ckpoint_file_name, 'w'):
        os.chmod(ckpoint_file_name, stat.S_IWUSR | stat.S_IRUSR)

    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 1

    ckp_mgr.remove_ckpoint_file(ckpoint_file_name)
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 0
    assert not os.path.exists(ckpoint_file_name)

    another_file_name = os.path.join(_cur_dir, './test2.ckpt')
    another_file_name = os.path.realpath(another_file_name)
    with open(another_file_name, 'w'):
        os.chmod(another_file_name, stat.S_IWUSR | stat.S_IRUSR)

    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 1
    ckp_mgr.remove_oldest_ckpoint_file()
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "test")
    assert ckp_mgr.ckpoint_num == 0
    assert not os.path.exists(another_file_name)

    # test keep_one_ckpoint_per_minutes
    file1 = os.path.realpath(os.path.join(_cur_dir, './time_file1.ckpt'))
    file2 = os.path.realpath(os.path.join(_cur_dir, './time_file2.ckpt'))
    file3 = os.path.realpath(os.path.join(_cur_dir, './time_file3.ckpt'))
    with open(file1, 'w'):
        os.chmod(file1, stat.S_IWUSR | stat.S_IRUSR)
    with open(file2, 'w'):
        os.chmod(file2, stat.S_IWUSR | stat.S_IRUSR)
    with open(file3, 'w'):
        os.chmod(file3, stat.S_IWUSR | stat.S_IRUSR)
    time1 = time.time()
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "time_file")
    assert ckp_mgr.ckpoint_num == 3
    ckp_mgr.keep_one_ckpoint_per_minutes(1, time1)
    ckp_mgr.update_ckpoint_filelist(_cur_dir, "time_file")
    assert ckp_mgr.ckpoint_num == 1
    if os.path.exists(_cur_dir + '/time_file1.ckpt'):
        os.chmod(_cur_dir + '/time_file1.ckpt', stat.S_IWRITE)
        os.remove(_cur_dir + '/time_file1.ckpt')


def test_load_param_into_net_error_net():
    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(TypeError):
        load_param_into_net('', parameter_dict)


def test_load_param_into_net_error_dict():
    net = Net(10)
    with pytest.raises(TypeError):
        load_param_into_net(net, '')


def test_load_param_into_net_erro_dict_param():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = ''
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(TypeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_has_more_param():
    """ test_load_param_into_net_has_more_param """
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    two_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.w"] = two_param
    load_param_into_net(net, parameter_dict)
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 1


def test_load_param_into_net_param_type_and_shape_error():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7))), name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_param_type_error():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.int32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net_param_shape_error():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7,)), dtype=mstype.int32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    with pytest.raises(RuntimeError):
        load_param_into_net(net, parameter_dict)


def test_load_param_into_net():
    net = Net(10)
    net.init_parameters_data()
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 0

    parameter_dict = {}
    one_param = Parameter(Tensor(np.ones(shape=(64, 3, 7, 7)), dtype=mstype.float32),
                          name="conv1.weight")
    parameter_dict["conv1.weight"] = one_param
    load_param_into_net(net, parameter_dict)
    assert net.conv1.weight.default_input.asnumpy()[0][0][0][0] == 1


def test_exec_save_checkpoint():
    net = Net()
    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = Momentum(net.trainable_params(), 0.0, 0.9, 0.0001, 1024)

    loss_net = WithLossCell(net, loss)
    train_network = TrainOneStepCell(loss_net, opt)
    _exec_save_checkpoint(train_network, ckpoint_file_name="./new_ckpt.ckpt")

    load_checkpoint("new_ckpt.ckpt")


def test_load_checkpoint_empty_file():
    os.mknod("empty.ckpt")
    with pytest.raises(ValueError):
        load_checkpoint("empty.ckpt")


class MYNET(nn.Cell):
    """ NET definition """

    def __init__(self):
        super(MYNET, self).__init__()
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


@non_graph_engine
def test_export():
    net = MYNET()
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    export(net, input_data, file_name="./me_export.pb", file_format="GEIR")


class BatchNormTester(nn.Cell):
    "used to test exporting network in training mode in onnx format"

    def __init__(self, num_features):
        super(BatchNormTester, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def construct(self, x):
        return self.bn(x)


class DepthwiseConv2dAndReLU6(nn.Cell):
    "Net for testing DepthwiseConv2d and ReLU6"

    def __init__(self, input_channel, kernel_size):
        super(DepthwiseConv2dAndReLU6, self).__init__()
        weight_shape = [1, input_channel, kernel_size, kernel_size]
        from mindspore.common.initializer import initializer
        self.weight = Parameter(initializer('ones', weight_shape), name='weight')
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=1, kernel_size=(kernel_size, kernel_size))
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        x = self.depthwise_conv(x, self.weight)
        x = self.relu6(x)
        return x


def test_batchnorm_train_onnx_export():
    input = Tensor(np.ones([1, 3, 32, 32]).astype(np.float32) * 0.01)
    net = BatchNormTester(3)
    net.set_train()
    if not net.training:
        raise ValueError('netowrk is not in training mode')
    export(net, input, file_name='batch_norm.onnx', file_format='ONNX')
    if not net.training:
        raise ValueError('netowrk is not in training mode')


class LeNet5(nn.Cell):
    """LeNet5 definition"""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_lenet5_onnx_export():
    input = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    net = LeNet5()
    export(net, input, file_name='lenet5.onnx', file_format='ONNX')


class DefinedNet(nn.Cell):
    """simple Net definition with maxpoolwithargmax."""

    def __init__(self, num_classes=10):
        super(DefinedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, weight_init="zeros")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = P.MaxPoolWithArgmax(padding="same", ksize=2, strides=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(int(56 * 56 * 64), num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, argmax = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def test_net_onnx_maxpoolwithargmax_export():
    input = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32) * 0.01)
    net = DefinedNet()
    export(net, input, file_name='definedNet.onnx', file_format='ONNX')


@run_on_onnxruntime
def test_lenet5_onnx_load_run():
    onnx_file = 'lenet5.onnx'

    input = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    net = LeNet5()
    export(net, input, file_name=onnx_file, file_format='ONNX')

    import onnx
    import onnxruntime as ort

    print('--------------------- onnx load ---------------------')
    # Load the ONNX model
    model = onnx.load(onnx_file)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    g = onnx.helper.printable_graph(model.graph)
    print(g)

    print('------------------ onnxruntime run ------------------')
    ort_session = ort.InferenceSession(onnx_file)
    input_map = {'x': input.asnumpy()}
    # provide only input x to run model
    outputs = ort_session.run(None, input_map)
    print(outputs[0])
    # overwrite default weight to run model
    for item in net.trainable_params():
        input_map[item.name] = np.ones(item.default_input.asnumpy().shape, dtype=np.float32)
    outputs = ort_session.run(None, input_map)
    print(outputs[0])


@run_on_onnxruntime
def test_depthwiseconv_relu6_onnx_load_run():
    onnx_file = 'depthwiseconv_relu6.onnx'
    input_channel = 3
    input = Tensor(np.ones([1, input_channel, 32, 32]).astype(np.float32) * 0.01)
    net = DepthwiseConv2dAndReLU6(input_channel, kernel_size=3)
    export(net, input, file_name=onnx_file, file_format='ONNX')

    import onnx
    import onnxruntime as ort

    print('--------------------- onnx load ---------------------')
    # Load the ONNX model
    model = onnx.load(onnx_file)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    g = onnx.helper.printable_graph(model.graph)
    print(g)

    print('------------------ onnxruntime run ------------------')
    ort_session = ort.InferenceSession(onnx_file)
    input_map = {'x': input.asnumpy()}
    # provide only input x to run model
    outputs = ort_session.run(None, input_map)
    print(outputs[0])
    # overwrite default weight to run model
    for item in net.trainable_params():
        input_map[item.name] = np.ones(item.default_input.asnumpy().shape, dtype=np.float32)
    outputs = ort_session.run(None, input_map)
    print(outputs[0])


def teardown_module():
    files = ['parameters.ckpt', 'new_ckpt.ckpt', 'lenet5.onnx', 'batch_norm.onnx', 'empty.ckpt']
    for item in files:
        file_name = './' + item
        if not os.path.exists(file_name):
            continue
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)
