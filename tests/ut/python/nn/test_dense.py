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
""" test nn.Dense """
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore.common.api import _executor
import mindspore.context as context
from mindspore import Tensor
from ..ut_filter import non_graph_engine


def test_dense_none():
    with pytest.raises(TypeError):
        nn.Dense(3, 2, None, None)


@non_graph_engine
def test_dense_str_activation():
    dense = nn.Dense(1, 1, activation='relu')
    assert isinstance(dense.activation, nn.ReLU)

    input_data = Tensor(np.random.randint(0, 255, [1, 1]).astype(np.float32))
    dense.construct(input_data)


def test_dense_weight_error():
    dim_error = Tensor(np.array([[[0.1], [0.3], [0.6]], [[0.4], [0.5], [0.2]]]))
    with pytest.raises(ValueError):
        nn.Dense(3, 2, dim_error)

    shape_error = Tensor(np.array([[0.1, 0.3, 0.6], [0.4, 0.5, 0.2]]))
    with pytest.raises(ValueError):
        nn.Dense(2, 2, shape_error)
    with pytest.raises(ValueError):
        nn.Dense(3, 3, shape_error)


def test_dense_bias_error():
    dim_error = Tensor(np.array([[0.5, 0.3]]))
    with pytest.raises(ValueError):
        nn.Dense(3, 2, bias_init=dim_error)

    shape_error = Tensor(np.array([0.5, 0.3, 0.4]))
    with pytest.raises(ValueError):
        nn.Dense(3, 2, bias_init=shape_error)


def test_dense_channels_error():
    with pytest.raises(ValueError):
        nn.Dense(3, 0)

    with pytest.raises(ValueError):
        nn.Dense(-1, 2)


class Net(nn.Cell):
    """ Net definition """
    def __init__(self,
                 input_channels,
                 output_channels,
                 weight='normal',
                 bias='zeros',
                 has_bias=True,
                 activation=''):
        super(Net, self).__init__()
        self.dense = nn.Dense(input_channels,
                              output_channels,
                              weight,
                              bias,
                              has_bias,
                              activation=activation)

    def construct(self, input_x):
        return self.dense(input_x)


def test_compile():
    """ test_compile """
    # has bias
    weight = Tensor(np.random.randint(0, 255, [8, 64]).astype(np.float32))
    bias = Tensor(np.random.randint(0, 255, [8]).astype(np.float32))
    net = Net(64, 8, weight=weight, bias=bias)
    input_data = Tensor(np.random.randint(0, 255, [128, 64]).astype(np.float32))
    _executor.compile(net, input_data)

    # training
    net_train = Net(64, 8, weight=weight, bias=bias)
    net_train.set_train()
    _executor.compile(net_train, input_data)


def test_compile_2():
    """ test_compile_2 """
    # no bias
    weight = Tensor(np.random.randint(0, 255, [8, 64]).astype(np.float32))
    net = Net(64, 8, weight=weight, has_bias=False)
    input_data = Tensor(np.random.randint(0, 255, [128, 64]).astype(np.float32))
    _executor.compile(net, input_data)

    # training
    net_train = Net(64, 8, weight=weight, has_bias=False)
    net_train.set_train()
    _executor.compile(net_train, input_data)


def test_compile_3():
    """ test_compile_3 """
    # test for Graph mode
    # has bias
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(128, 10)
    input_data = Tensor(np.random.randint(0, 255, [128, 128]).astype(np.float32))
    _executor.compile(net, input_data)

    # training
    net_train = Net(128, 10)
    net_train.set_train()
    _executor.compile(net_train, input_data)


def test_compile_4():
    """ test_compile_4 """
    # test for Graph mode
    # no bias
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(128, 10, has_bias=False)
    input_data = Tensor(np.random.randint(0, 255, [128, 128]).astype(np.float32))
    _executor.compile(net, input_data)

    # training
    net_train = Net(128, 10, has_bias=False)
    net_train.set_train()
    _executor.compile(net_train, input_data)
