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
"""test_mix_precision"""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common import ParameterTuple
from mindspore.common.api import _executor
from mindspore.common.parameter import Parameter
from mindspore.nn import Momentum
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.context import ParallelMode
from tests.ops_common import convert
from ....train_step_wrap import train_step_with_loss_warp


class LeNet5(nn.Cell):
    """LeNet5"""

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


class NetForConcat(nn.Cell):
    def __init__(self):
        super(NetForConcat, self).__init__()
        self.concat = P.Concat()
        self.x1 = Tensor(np.zeros([1, 10]).astype(np.float32))
        self.x2 = Parameter(Tensor(np.zeros([1, 10]).astype(np.float32)), name='x2')

    def construct(self, x0):
        return self.concat((x0, self.x1, self.x2))


def test_add_cast_flag():
    predict = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = LeNet5()
    net.to_float(mstype.float16)
    net.fc3.to_float(mstype.float32)
    net = train_step_with_loss_warp(net)
    net.set_train()
    net(predict, label)


def test_add_cast_flag_tensor():
    x1 = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = NetForConcat()
    net.add_flags_recursive(fp16=True)
    net.set_train()
    net(x1)


def test_on_momentum():
    predict = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = LeNet5()
    net = train_step_with_loss_warp(net).to_float(mstype.float16)
    net.set_train()
    net(predict, label)


def test_data_parallel_with_cast():
    """test_data_parallel_with_cast"""
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=8)
    predict = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = LeNet5()
    net.to_float(mstype.float16)
    net.fc3.to_float(mstype.float32)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    net = WithLossCell(net, loss_fn)
    net = TrainOneStepCell(net, optimizer)

    _executor.compile(net, predict, label)
    context.reset_auto_parallel_context()


class NetForPReLU(nn.Cell):
    def __init__(self):
        super(NetForPReLU, self).__init__()
        self.prelu = nn.PReLU()

    def construct(self, x):
        return self.prelu(x)


def test_nn_prelu():
    x = Tensor(np.ones([1, 16, 10, 10]).astype(np.float32) * 0.01)
    net = NetForPReLU().set_train()
    net.add_flags_recursive(fp16=True)
    _executor.compile(net, x)


class NetForCast(nn.Cell):
    def __init__(self):
        super(NetForCast, self).__init__()
        self.x1 = Tensor(1.0, mstype.float32)
        self.x2 = Parameter(Tensor(np.zeros([1, 10]).astype(np.float32)), name='x2')

    def construct(self, x0):
        x = self.x1 * x0 * self.x2
        return x


def test_cast():
    x = Tensor(np.ones([1, 16, 10, 10]).astype(np.float32) * 0.01)
    net = NetForCast()
    net.add_flags_recursive(fp16=True)
    net(x)


class IRBlockZ(nn.Cell):
    def __init__(self, inplanes, planes):
        super(IRBlockZ, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, pad_mode="same", group=1, has_bias=False,
                               dilation=1)
        self.act_layer = nn.PReLU(planes)

    def construct(self, x):
        out = self.conv1(x)
        return self.act_layer(out)


class GetParamGrad(nn.Cell):
    def __init__(self, network):
        super(GetParamGrad, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)

    def construct(self, data, sens):
        weights = self.weights
        return self.grad(self.network, weights)(data, sens)


def test_grad_conv_prelu():
    shapes = [[64, 64, 112, 112]]
    outshape = [[64, 64, 112, 112]]
    net = IRBlockZ(inplanes=64, planes=64).add_flags_recursive(fp16=True)
    inputs = [convert(shp, dtype=np.float16) for shp in shapes]
    sens_shape = outshape[0]
    sens = convert(sens_shape, dtype=np.float16)
    all_inputs = inputs + [sens]
    net = GetParamGrad(net)
    net.set_train()
    net(*all_inputs)


def test_dict_cast():
    class FirstNet(nn.Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.net = SecondNet()
            self.sub = P.Sub()

        def construct(self, tensor_a, tensor_b):
            a = F.mixed_precision_cast(mstype.float16, tensor_a)
            b = F.mixed_precision_cast(mstype.float16, tensor_b)
            c = self.sub(a, b)
            dictionary = {"key": a}
            result = self.net(c, key1=a, key2=dictionary)
            return result

    class SecondNet(nn.Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
            self.add = P.Add()

        def construct(self, tensor_c, **kwargs):
            d = F.mixed_precision_cast(mstype.float16, tensor_c)
            dict_cast = F.mixed_precision_cast(mstype.float16, kwargs)
            e = self.add(d, dict_cast["key1"])
            f = self.add(e, dict_cast["key2"]["key"])
            return f

    x = Tensor(np.array([1, 2.5, 3.5]), mstype.float32)
    y = Tensor(np.array([4, 5.5, 6.5]), mstype.float32)
    net = FirstNet()
    net(x, y)


def test_kwarg_cast():
    class FirstNet(nn.Cell):
        def __init__(self):
            super(FirstNet, self).__init__()
            self.net = SecondNet().add_flags_recursive(fp16=True)
            self.add = P.Add()

        def construct(self, tensor_a, tensor_b):
            tensor_c = self.add(tensor_a, tensor_b)
            dictionary = {"key": tensor_a}
            result = self.net(key1=tensor_c, key2=dictionary)
            return result

    class SecondNet(nn.Cell):
        def __init__(self):
            super(SecondNet, self).__init__()
            self.add = P.Add()

        def construct(self, key1=1, key2=2):
            tensor_d = self.add(key1, key2["key"])
            return tensor_d

    x = Tensor(np.array([1, 2.5, 3.5]), mstype.float32)
    y = Tensor(np.array([4, 5.5, 6.5]), mstype.float32)
    net = FirstNet()
    net(x, y)
