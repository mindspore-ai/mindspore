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
""" test_initializer """
import math
from functools import reduce
import numpy as np
import pytest as py
from scipy import stats

import mindspore as ms
import mindspore.common.initializer as init
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Conv2d
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine


# pylint: disable=W0212
# W0212: protected-access

class InitTwo(init.Initializer):
    """Initialize the array to two."""

    def _initialize(self, arr):
        init._assignment(arr, 2)


def _check_value(tensor, value_min, value_max):
    nd = tensor.asnumpy()
    for ele in nd.flatten():
        if value_min <= ele <= value_max:
            continue
        raise ValueError('value_min = %d, ele = %d, value_max = %d'
                         % (value_min, ele, value_max))


def _check_uniform(tensor, boundary_a, boundary_b):
    samples = tensor.asnumpy().reshape((-1))
    _, p = stats.kstest(samples, 'uniform', (boundary_a, (boundary_b - boundary_a)))
    print("p-value is %f" % p)
    return p > 0.0001


def test_init_Initializer():
    tensor = init.initializer(InitTwo(), [2, 2], ms.int32)
    assert tensor.shape == (2, 2)
    _check_value(tensor.init_data(), 2, 2)


def test_init_tensor():
    tensor = ms.Tensor(np.zeros([1, 2, 3]))
    tensor = init.initializer(tensor, [1, 2, 3], ms.float32)
    assert tensor.shape == (1, 2, 3)


def test_init_zero_default_dtype():
    tensor = init.initializer(init.Zero(), [2, 2])
    assert tensor.dtype == ms.float32
    _check_value(tensor.init_data(), 0, 0)


def test_init_zero():
    tensor = init.initializer(init.Zero(), [2, 2], ms.float32)
    _check_value(tensor.init_data(), 0, 0)


def test_init_zero_alias_default_dtype():
    tensor = init.initializer('zeros', [1, 2])
    assert tensor.dtype == ms.float32
    _check_value(tensor.init_data(), 0, 0)


def test_init_zero_alias():
    tensor = init.initializer('zeros', [1, 2], ms.float32)
    _check_value(tensor.init_data(), 0, 0)


def test_init_one():
    tensor = init.initializer(init.One(), [2, 2], ms.float32)
    _check_value(tensor.init_data(), 1, 1)


def test_init_one_alias():
    tensor = init.initializer('ones', [1, 2], ms.float32)
    _check_value(tensor.init_data(), 1, 1)


def test_init_constant():
    tensor = init.initializer(init.Constant(1), [2, 2], ms.float32)
    _check_value(tensor.init_data(), 1, 1)


def test_init_uniform():
    scale = 10
    tensor = init.initializer(init.Uniform(scale=scale), [5, 4], ms.float32)
    _check_value(tensor.init_data(), -scale, scale)


def test_init_uniform_alias():
    scale = 100
    tensor = init.initializer('uniform', [5, 4], ms.float32)
    _check_value(tensor.init_data(), -scale, scale)


def test_init_normal():
    tensor = init.initializer(init.Normal(), [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'Normal init failed!'


def test_init_truncated_normal():
    tensor = init.initializer(init.TruncatedNormal(), [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'TruncatedNormal init failed!'


def test_init_normal_alias():
    tensor = init.initializer('normal', [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'Normal init failed!'


def test_init_truncatednormal_alias():
    tensor = init.initializer('truncatednormal', [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'TruncatedNormal init failed!'


def test_init_abnormal():
    with py.raises(TypeError):
        init.initializer([''], [5, 4], ms.float32)


def test_initializer_reinit():
    weights = init.initializer("XavierUniform", shape=(10, 1, 10, 10), dtype=ms.float16)
    assert isinstance(weights, Tensor), 'XavierUniform init failed!'


def test_init_xavier_uniform():
    """ test_init_xavier_uniform """
    gain = 1.2
    tensor1 = init.initializer(init.XavierUniform(gain=gain), [20, 22], ms.float32).init_data()
    tensor2 = init.initializer(init.XavierUniform(), [20, 22], ms.float32).init_data()
    tensor3 = init.initializer(init.XavierUniform(gain=gain), [20, 22, 5, 5], ms.float32).init_data()
    tensor4 = init.initializer(init.XavierUniform(), [20, 22, 5, 5], ms.float32).init_data()
    tensor5 = init.initializer('xavier_uniform', [20, 22, 5, 5], ms.float32).init_data()
    tensor6 = init.initializer('xavier_uniform', [20, 22], ms.float32).init_data()
    tensor_dict = {tensor1: gain, tensor2: None, tensor3: gain, tensor4: None, tensor5: None, tensor6: None}

    for tensor, gain_value in tensor_dict.items():
        if gain_value is None:
            gain_value = 1
        shape = tensor.asnumpy().shape
        if len(shape) > 2:
            s = reduce(lambda x, y: x * y, shape[2:])
        else:
            s = 1
        n_in = shape[1] * s
        n_out = shape[0] * s
        std = gain_value * math.sqrt(2 / (n_in + n_out))
        boundary = std * math.sqrt(3)
        assert _check_uniform(tensor, -boundary, boundary)


def test_init_xavier_uniform_error():
    with py.raises(ValueError):
        init.initializer(init.XavierUniform(), [6], ms.float32).init_data()


def test_init_he_uniform():
    """ test_init_he_uniform """
    tensor1 = init.initializer(init.HeUniform(), [20, 22], ms.float32)
    tensor2 = init.initializer(init.HeUniform(), [20, 22, 5, 5], ms.float32)
    tensor3 = init.initializer('he_uniform', [20, 22, 5, 5], ms.float32)
    tensor4 = init.initializer('he_uniform', [20, 22], ms.float32)
    tensors = [tensor1.init_data(), tensor2.init_data(), tensor3.init_data(), tensor4.init_data()]

    for tensor in tensors:
        shape = tensor.asnumpy().shape
        if len(shape) > 2:
            s = reduce(lambda x, y: x * y, shape[2:])
        else:
            s = 1
        n_in = shape[1] * s
        std = math.sqrt(2 / n_in)
        boundary = std * math.sqrt(3)
        assert _check_uniform(tensor, -boundary, boundary)


def test_init_he_uniform_error():
    with py.raises(ValueError):
        init.initializer(init.HeUniform(), [6], ms.float32).init_data()


def test_conv2d_abnormal_kernel_negative():
    kernel = np.random.randn(64, 3, 7, 7).astype(np.float32)
    with py.raises(ValueError):
        ms.Model(
            Conv2d(in_channels=3, out_channels=64, kernel_size=-7, stride=3,
                   padding=0, weight_init=ms.Tensor(kernel)))


@non_graph_engine
def test_conv2d_abnormal_kernel_normal():
    kernel = np.random.randn(64, 3, 7, 7).astype(np.float32)
    input_data = np.random.randn(32, 3, 224, 112).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE)
    model = ms.Model(
        Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3,
               padding=0, weight_init=ms.Tensor(kernel)))
    model.predict(ms.Tensor(input_data))


@non_graph_engine
def test_conv2d_abnormal_kernel_truncated_normal():
    input_data = init.initializer(init.TruncatedNormal(), [64, 3, 7, 7], ms.float32).init_data()
    context.set_context(mode=context.GRAPH_MODE)
    model = ms.Model(
        Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3,
               padding=0, weight_init="truncatednormal"))
    model.predict(input_data)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.t1 = Parameter(init.initializer('uniform', [5, 4], ms.float32), name="w1")
        self.t2 = Parameter(init.initializer(init.TruncatedNormal(), [5, 4], ms.float32), name="w2")

    def construct(self, x):
        z = self.add(x, self.t1)
        z = self.add(z, self.t2)
        return z


def test_weight_shape():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    a = np.arange(20).reshape(5, 4)
    t = Tensor(a, dtype=ms.float32)
    net = Net()
    out = net(t)
    print(out)
