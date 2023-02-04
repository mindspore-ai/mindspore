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
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops.operations._rl_inner_ops import GRUV2


class Net(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional):
        super().__init__()
        self.gru_nn = nn.GRU(input_size, hidden_size, num_layers, has_bias, False, 0.0, bidirectional)

    def construct(self, x, h, seq_lengths):
        output, hy = self.gru_nn(x, h, seq_lengths)
        return output, hy


class NetGruV2(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, has_bias, weights, is_train):
        super(NetGruV2, self).__init__()
        self.gruv2 = GRUV2(input_size, hidden_size, num_layers, has_bias, False, 0.0, is_train)
        self.weights = weights

    def construct(self, x, h_0, seq_len):
        output, h_n, _, _ = self.gruv2(x, h_0, self.weights.astype(x.dtype), seq_len)
        return output, h_n


def get_weights_from_gru(gru_nn, has_bias):
    if has_bias:
        weights = ops.concat((
            gru_nn.w_ih_list[0].view(-1, 1, 1),
            gru_nn.w_hh_list[0].view(-1, 1, 1),
            gru_nn.b_ih_list[0].view(-1, 1, 1),
            gru_nn.b_hh_list[0].view(-1, 1, 1)
        ))
    else:
        weights = ops.concat((
            gru_nn.w_ih_list[0].view(-1, 1, 1),
            gru_nn.w_hh_list[0].view(-1, 1, 1),
        ))
    return weights


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("is_train", [True, False])
@pytest.mark.parametrize("dtype", [ms.float16, ms.float32])
def test_gruv2_op(has_bias, is_train, dtype):
    """
    Feature: test GRUV2
    Description: num_layers=1, bidirectional=False
    Expectation: the result is equal to nn.GRU.
    """
    batch_size = 3
    max_seq_length = 5
    input_size = 10
    hidden_size = 3
    num_layers = 1
    bidirectional = False
    num_directions = 2 if bidirectional else 1
    seq_lengths = Tensor([5, 3, 2], ms.int32)

    x = Tensor(np.random.normal(0.0, 1.0, (max_seq_length, batch_size, input_size)), dtype)
    h0 = Tensor(np.random.normal(0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), dtype)
    net = Net(input_size, hidden_size, num_layers, has_bias, bidirectional).set_train(is_train)
    weights = get_weights_from_gru(net.gru_nn, has_bias)
    gruv2_net = NetGruV2(input_size, hidden_size, num_layers, has_bias, weights, is_train)
    expect_output, expect_hy = net(x, h0, seq_lengths)
    me_output, me_hy = gruv2_net(x, h0, seq_lengths)

    rtol, atol = (1e-3, 1e-3) if dtype == ms.float16 else (1e-4, 1e-4)
    assert np.allclose(me_output.asnumpy(), expect_output.asnumpy(), rtol, atol)
    assert np.allclose(me_hy.asnumpy(), expect_hy.asnumpy(), rtol, atol)


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
    x = Tensor(np.random.normal(0.0, 1.0, (max_seq_length, batch_size, input_size)), ms.float64)
    h0 = Tensor(np.random.normal(0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), ms.float64)
    weights = Tensor(np.random.normal(0.0, 1.0, (3 * hidden_size * (input_size + hidden_size), 1, 1)), ms.float64)
    net = NetGruV2(input_size, hidden_size, num_layers, False, weights, False)
    with pytest.raises(TypeError):
        net(x, h0, seq_lengths)
