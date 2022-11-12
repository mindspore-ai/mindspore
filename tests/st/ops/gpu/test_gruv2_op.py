# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.gru_nn = nn.GRU(input_size, hidden_size, num_layers, True, True, 0.0, bidirectional)

    def construct(self, x, h, seq_lengths):
        _, hy = self.gru_nn(x, h, seq_lengths)
        return hy


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_train", [True, False])
def test_gruv2_op_float32_1(is_train):
    """
    Feature: test GRUV2 with using float32
    Description: num_layers=1, bidirectional=False
    Expectation: the result match with expect.
    """
    batch_size = 3
    max_seq_length = 5
    input_size = 10
    hidden_size = 3
    num_layers = 1
    bidirectional = False
    num_directions = 2 if bidirectional else 1
    seq_lengths = Tensor([5, 3, 2], ms.int32)

    np.random.seed(1)
    x = Tensor(np.random.normal(0.0, 1.0, (batch_size, max_seq_length, input_size)), ms.float32)
    h0 = Tensor(np.random.normal(0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), ms.float32)
    net = Net(input_size, hidden_size, num_layers, bidirectional)
    net.set_train(is_train)
    me_hy = net(x, h0, seq_lengths).asnumpy()
    expect_hy = np.array([[[0.23690273, -0.42312058, 0.2012992],
                           [0.5544311, -0.28084755, -0.03353014],
                           [0.12614538, -0.26933774, 0.11727069]]], np.float32)
    assert np.allclose(me_hy, expect_hy, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_train", [True, False])
def test_gruv2_op_float32_2(is_train):
    """
    Feature: test GRUV2 with using float32
    Description: num_layers=3, bidirectional=True
    Expectation: the result match with expect.
    """
    batch_size = 3
    max_seq_length = 5
    input_size = 10
    hidden_size = 1
    num_layers = 1
    bidirectional = True
    num_directions = 2 if bidirectional else 1
    seq_lengths = Tensor([5, 3, 2], ms.int32)

    np.random.seed(4)
    x = Tensor(np.random.normal(0.0, 1.0, (batch_size, max_seq_length, input_size)), ms.float32)
    h0 = Tensor(np.random.normal(0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), ms.float32)
    net = Net(input_size, hidden_size, num_layers, bidirectional)
    net.set_train(is_train)
    me_hy = net(x, h0, seq_lengths).asnumpy()
    expect_hy = np.array([[[0.32341897], [0.83405745], [0.22347865]], [[-0.40905663], [-0.8938196], [-0.8207804]]],
                         np.float32)
    assert np.allclose(me_hy, expect_hy, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_train", [True, False])
def test_gruv2_op_float16(is_train):
    """
    Feature: test GRUV2 with using float16
    Description: num_layers=1, bidirectional=False
    Expectation: the result match with expect.
    """
    batch_size = 3
    max_seq_length = 5
    input_size = 10
    hidden_size = 3
    num_layers = 1
    bidirectional = False
    num_directions = 2 if bidirectional else 1
    seq_lengths = Tensor([5, 3, 2], ms.int32)

    np.random.seed(1)
    x = Tensor(np.random.normal(0.0, 1.0, (batch_size, max_seq_length, input_size)), ms.float16)
    h0 = Tensor(np.random.normal(0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), ms.float16)
    net = Net(input_size, hidden_size, num_layers, bidirectional)
    net.set_train(is_train)
    me_hy = net(x, h0, seq_lengths).asnumpy()
    expect_hy = np.array([[[0.2368, -0.4233, 0.2017], [0.5547, -0.281, -0.03323], [0.1263, -0.2693, 0.1175]]],
                         np.float16)
    assert np.allclose(me_hy, expect_hy, 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gruv2_op_float64_exception():
    """
    Feature: test GRUV2 with using float64
    Description: Using float64
    Expectation: Raise TypeError.
    """
    batch_size = 3
    max_seq_length = 5
    input_size = 10
    hidden_size = 3
    num_layers = 1
    bidirectional = False
    num_directions = 2 if bidirectional else 1
    seq_lengths = Tensor([5, 3, 2], ms.int32)

    np.random.seed(1)
    x = Tensor(np.random.normal(0.0, 1.0, (batch_size, max_seq_length, input_size)), ms.float64)
    h0 = Tensor(np.random.normal(0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), ms.float64)
    net = Net(input_size, hidden_size, num_layers, bidirectional)
    with pytest.raises(TypeError):
        net(x, h0, seq_lengths)
