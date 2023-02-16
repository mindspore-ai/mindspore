# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.ops.operations._rl_inner_ops as rl_ops
import mindspore.ops.operations._grad_ops as grad_ops
from mindspore import context, Tensor
from mindspore.common.parameter import ParameterTuple
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import composite as c


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_gru_grad(mode):
    """
    Feature: test gru_grad cpu operation.
    Description: test gru_grad cpu operation.
    Expectation: no exception.
    """
    input_size = 10
    hidden_size = 2
    num_layers = 1
    max_seq_len = 5
    batch_size = 2

    context.set_context(mode=mode)
    net = rl_ops.GRUV2(input_size, hidden_size, num_layers, True, False, 0.0)
    input_tensor = Tensor(
        np.ones([max_seq_len, batch_size, input_size]).astype(np.float32))
    h0 = Tensor(
        np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
    w = Tensor(np.ones([84, 1, 1]).astype(np.float32))
    seq_lengths = Tensor(np.array([4, 3]).astype(np.int32))
    output, hn, out1, _ = net(input_tensor, h0, w, seq_lengths)
    grad_net = grad_ops.GRUV2Grad(
        input_size, hidden_size, num_layers, True, False, 0.0)
    dx, dh, dw = grad_net(input_tensor, h0, w, seq_lengths,
                          output, hn, output, hn, out1)
    print("dx:", dx)
    print("dh:", dh)
    print("dw:", dw)


class GradOfAllInputsAndParams(nn.Cell):
    def __init__(self, network, sens_param):
        super().__init__()
        self.grad = c.GradOperation(
            get_all=True, get_by_list=True, sens_param=sens_param)
        self.network = network
        self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        gout = self.grad(self.network, self.params)(*inputs)
        return gout


class NetGruV2(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, has_bias, weights, is_train):
        super(NetGruV2, self).__init__()
        self.gruv2 = rl_ops.GRUV2(
            input_size, hidden_size, num_layers, has_bias, False, 0.0, is_train)
        self.weights = weights

    def construct(self, x, h_0, seq_len):
        return self.gruv2(
            x, h_0, self.weights.astype(x.dtype), seq_len)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("is_train", [True, False])
def test_gru_backward(has_bias, is_train):
    """
    Feature: test GRUV2 backward.
    Description: test gru_grad cpu operation.
    Expectation: no exception.
    """
    batch_size = 3
    max_seq_length = 5
    input_size = 10
    hidden_size = 3
    num_layers = 1
    num_directions = 1
    seq_lengths = Tensor([5, 3, 2], ms.int32)
    dtype = ms.float32

    x = Tensor(np.random.normal(
        0.0, 1.0, (max_seq_length, batch_size, input_size)), dtype)
    h0 = Tensor(np.random.normal(
        0.0, 1.0, (num_layers * num_directions, batch_size, hidden_size)), dtype)
    weight_size = 135 if has_bias else 117
    weights = Tensor(np.ones([weight_size, 1, 1]).astype(np.float32))

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    gru_v2_net = NetGruV2(input_size, hidden_size,
                          num_layers, has_bias, weights, is_train)
    grad_net_inp = GradOfAllInputsAndParams(gru_v2_net, sens_param=False)
    grad_net_inp.set_train()
    out_grad, _ = grad_net_inp(x, h0, seq_lengths)
    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_gru_v2_net = NetGruV2(input_size, hidden_size,
                                   num_layers, has_bias, weights, is_train)
    pynative_grad_net_inp = GradOfAllInputsAndParams(
        pynative_gru_v2_net, sens_param=False)
    pynative_grad_net_inp.set_train()
    py_native_out_grad, _ = pynative_grad_net_inp(x, h0, seq_lengths)

    assert np.allclose(out_grad[0].asnumpy(),
                       py_native_out_grad[0].asnumpy(), 0.001, 0.001)
    assert np.allclose(out_grad[1].asnumpy(),
                       py_native_out_grad[1].asnumpy(), 0.001, 0.001)
