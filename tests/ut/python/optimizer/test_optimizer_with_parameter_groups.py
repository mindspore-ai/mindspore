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
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.common.tensor import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum, SGD, RMSProp, Adam
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


class LeNet5(nn.Cell):
    """ LeNet5 definition """
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


def test_group_lr():
    inputs = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))

    net = LeNet5()
    conv_lr = 0.8
    default_lr = 0.1
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': no_conv_params},
                    {'params': conv_params, 'lr': conv_lr},
                    {'order_params': net.trainable_params()}]
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    opt = Momentum(group_params, learning_rate=default_lr, momentum=0.9)
    assert opt.is_group is True
    assert opt.is_group_lr is True
    assert opt.dynamic_lr is False
    assert opt.is_group_params_ordered is True
    for lr, param, order_param in zip(opt.learning_rate, opt.parameters, net.trainable_params()):
        if 'conv' in param.name:
            assert np.all(lr.data.asnumpy() == Tensor(conv_lr, mstype.float32).asnumpy())
        else:
            assert np.all(lr.data.asnumpy() == Tensor(default_lr, mstype.float32).asnumpy())

        assert param.name == order_param.name

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, opt)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_group_dynamic_1():
    inputs = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))

    net = LeNet5()
    conv_lr = 0.8
    default_lr = (0.1, 0.2, 0.3)
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': no_conv_params},
                    {'params': conv_params, 'lr': conv_lr},
                    {'order_params': net.trainable_params()}]
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    opt = Momentum(group_params, learning_rate=default_lr, momentum=0.9)
    assert opt.is_group is True
    assert opt.dynamic_lr is True
    assert opt.is_group_params_ordered is True
    for lr, param, order_param in zip(opt.learning_rate, opt.parameters, net.trainable_params()):
        if 'conv' in param.name:
            assert np.all(lr.learning_rate.data.asnumpy() == \
                          Tensor(np.array([conv_lr] * 3).astype(np.float32)).asnumpy())
        else:
            assert np.all(lr.learning_rate.data.asnumpy() == \
                          Tensor(np.array(list(default_lr)).astype(np.float32)).asnumpy())

        assert param.name == order_param.name

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, opt)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_group_dynamic_2():
    inputs = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))

    net = LeNet5()
    conv_lr = (0.1, 0.2, 0.3)
    default_lr = 0.8
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'lr': conv_lr},
                    {'params': no_conv_params}]
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    opt = RMSProp(group_params, learning_rate=default_lr)
    assert opt.is_group is True
    assert opt.dynamic_lr is True
    for lr, param in zip(opt.learning_rate, opt.parameters):
        if 'conv' in param.name:
            assert np.all(lr.learning_rate.data.asnumpy() == \
                          Tensor(np.array(list(conv_lr)).astype(np.float32)).asnumpy())
        else:
            assert np.all(lr.learning_rate.data.asnumpy() == \
                          Tensor(np.array([default_lr] * 3).astype(np.float32)).asnumpy())

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, opt)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_group_dynamic_no_same_size():
    net = LeNet5()
    conv_lr = (0.1, 0.2, 0.3)
    default_lr = (0.1, 0.2)
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'lr': conv_lr},
                    {'params': no_conv_params}]
    with pytest.raises(ValueError):
        Momentum(group_params, learning_rate=default_lr, momentum=0.9)


def test_group_not_float_lr():
    net = LeNet5()
    conv_lr = np.array(1)
    default_lr = 0.3
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'lr': conv_lr},
                    {'params': no_conv_params}]
    with pytest.raises(TypeError):
        Momentum(group_params, learning_rate=default_lr, momentum=0.9)


def test_group_not_float_weight_decay():
    net = LeNet5()
    conv_weight_decay = np.array(1)
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': conv_weight_decay},
                    {'params': no_conv_params}]
    with pytest.raises(TypeError):
        Momentum(group_params, learning_rate=0.1, momentum=0.9)


def test_weight_decay():
    inputs = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))

    net = LeNet5()
    conv_weight_decay = 0.8
    default_weight_decay = 0.0
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': no_conv_params},
                    {'params': conv_params, 'weight_decay': conv_weight_decay},
                    {'order_params': net.trainable_params()}]
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    opt = SGD(group_params, learning_rate=0.1, weight_decay=default_weight_decay)
    assert opt.is_group is True
    assert opt.is_group_lr is False
    assert opt.is_group_params_ordered is True
    for weight_decay, decay_flags, param, order_param in zip(
            opt.weight_decay, opt.decay_flags, opt.parameters, net.trainable_params()):
        if 'conv' in param.name:
            assert abs(weight_decay.asnumpy() - conv_weight_decay) < 1.e-6
            assert decay_flags is True
        else:
            assert abs(weight_decay.asnumpy() - default_weight_decay) < 1.e-6
            assert decay_flags is False

        assert param.name == order_param.name

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, opt)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_group_repeat_param():
    net = LeNet5()
    conv_lr = 0.1
    default_lr = 0.3
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'lr': conv_lr},
                    {'params': conv_params, 'lr': default_lr},
                    {'params': no_conv_params}]
    with pytest.raises(RuntimeError):
        Adam(group_params, learning_rate=default_lr)


def test_get_lr_parameter_with_group():
    net = LeNet5()
    conv_lr = 0.1
    default_lr = 0.3
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'lr': conv_lr},
                    {'params': no_conv_params, 'lr': default_lr}]
    opt = SGD(group_params)
    assert opt.is_group_lr is True
    for param in opt.parameters:
        lr = opt.get_lr_parameter(param)
        if 'conv' in param.name:
            cur_name = 'learning_rate_group_' + '0'
        else:
            cur_name = 'learning_rate_group_' + '1'
        assert lr.name == cur_name

    lr_list = opt.get_lr_parameter(conv_params)
    for lr, param in zip(lr_list, conv_params):
        assert lr.name == 'learning_rate_group_' + '0'


def test_get_lr_parameter_with_order_group():
    net = LeNet5()
    conv_lr = 0.1
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'lr': conv_lr},
                    {'params': no_conv_params}]
    opt = SGD(group_params)
    assert opt.is_group_lr is True
    for param in opt.parameters:
        lr = opt.get_lr_parameter(param)
        if 'conv' in param.name:
            cur_name = 'learning_rate_group_' + '0'
        else:
            cur_name = 'learning_rate'
        assert lr.name == cur_name


def test_get_lr_parameter_with_no_group():
    net = LeNet5()
    conv_weight_decay = 0.8

    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': conv_weight_decay},
                    {'params': no_conv_params}]
    opt = SGD(group_params)
    assert opt.is_group_lr is False
    for param in opt.parameters:
        lr = opt.get_lr_parameter(param)
        assert lr.name == 'learning_rate'

    params_error = [1, 2, 3]
    with pytest.raises(TypeError):
        opt.get_lr_parameter(params_error)


def test_order_params_1():
    net = LeNet5()
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    bias_params = list(filter(lambda x: 'bias' in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01},
                    {'params': bias_params, 'lr': 0.01},
                    {'order_params': bias_params+conv_params}]
    opt = SGD(group_params, learning_rate=0.1, weight_decay=0.0)
    assert opt.is_group is True
    assert opt.is_group_lr is True
    assert opt.is_group_params_ordered is True
    for weight_decay, decay_flags, lr, param, order_param in zip(
            opt.weight_decay, opt.decay_flags, opt.learning_rate, opt.parameters, bias_params+conv_params):
        if 'conv' in param.name:
            assert np.all(lr.data.asnumpy() == Tensor(0.1, mstype.float32).asnumpy())
            assert abs(weight_decay.asnumpy() - 0.01) < 1.e-6
            assert decay_flags is True
        elif 'bias' in param.name:
            assert np.all(lr.data.asnumpy() == Tensor(0.01, mstype.float32).asnumpy())
            assert abs(weight_decay.asnumpy()) < 1.e-6
            assert decay_flags is False
        else:
            assert np.all(lr.data.asnumpy() == Tensor(0.1, mstype.float32).asnumpy())
            assert abs(weight_decay.asnumpy()) < 1.e-6
            assert decay_flags is False

        assert param.name == order_param.name
        if 'conv' in param.name:
            assert lr.name == 'learning_rate'
        elif 'bias' in param.name:
            assert lr.name == 'learning_rate_group_' + '1'



def test_order_params_2():
    net = LeNet5()
    conv_weight_decay = 0.01
    fc1_lr = (0.5, 0.4, 0.3)
    default_lr = 0.1
    default_wd = 0.0
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    fc1_params = list(filter(lambda x: 'fc1' in x.name, net.trainable_params()))
    group_params = [{'params': fc1_params, 'lr': fc1_lr},
                    {'params': conv_params, 'weight_decay': conv_weight_decay},
                    {'order_params': fc1_params+conv_params}]
    opt = SGD(group_params, learning_rate=default_lr, weight_decay=default_wd)
    assert opt.is_group is True
    assert opt.is_group_lr is True
    assert opt.is_group_params_ordered is True
    all_lr = opt.get_lr_parameter(fc1_params+conv_params)
    for weight_decay, decay_flags, lr, param, order_param in zip(
            opt.weight_decay, opt.decay_flags, all_lr, opt.parameters, fc1_params+conv_params):
        if 'conv' in param.name:
            assert np.all(lr.data.asnumpy() == Tensor(np.array([default_lr] * 3), mstype.float32).asnumpy())
            assert abs(weight_decay.asnumpy() - conv_weight_decay) < 1.e-6
            assert decay_flags is True
        elif 'fc1' in param.name:
            assert np.all(lr.data.asnumpy() == Tensor(fc1_lr, mstype.float32).asnumpy())
            assert abs(weight_decay.asnumpy() - default_wd) < 1.e-6
            assert decay_flags is False
        else:
            assert np.all(lr.data.asnumpy() == Tensor(np.array([default_lr] * 3), mstype.float32).asnumpy())
            assert abs(weight_decay.asnumpy() - default_wd) < 1.e-6
            assert decay_flags is False

        assert param.name == order_param.name
        if 'conv' in param.name:
            assert lr.name == 'learning_rate'
        elif 'fc1' in param.name:
            assert lr.name == 'learning_rate_group_' + '0'

def test_get_order_params_with_not_same():
    net = LeNet5()
    conv_weight_decay = 0.8

    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': conv_weight_decay},
                    {'order_params': no_conv_params}]
    with pytest.raises(ValueError):
        SGD(group_params)
