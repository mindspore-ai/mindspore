# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import mint, context, jit
from mindspore import Tensor
from tests.mark_utils import arg_mark


@jit(backend="ms_backend")
def backward_func(net, input_x):
    return ms.grad(net)(input_x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_conv1d_ext_default(mode):
    """
    Feature: Conv1d
    Description: Verify the result of specifying Conv1d customed para dtype.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    if mode == context.GRAPH_MODE:
        context.set_context(jit_config={"jit_level": "O0"})
    input_x = Tensor(np.linspace(0, 10, 1 * 2 * 2),
                     ms.float32).reshape(1, 2, 2)
    in_channels = 2
    out_channels = 2
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    bias = True
    padding_mode = 'zeros'
    dtype = ms.float32

    net = mint.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                         dtype=dtype)
    net.weight = ms.Parameter(ms.Tensor(np.ones((2, 2, 1)), dtype=dtype), name='weight')
    net.bias = ms.Parameter(ms.Tensor(np.ones((2,)), dtype=dtype), name='bias')

    out = net(input_x)
    input_grad = backward_func(net, input_x)
    expected_out = [[[7.6679688, 14.3359375],
                     [7.6679688, 14.3359375]]]
    expect_input_grad = [[[2., 2.],
                          [2., 2.]]]
    assert np.allclose(out.asnumpy(), expected_out, 1e-4, 1e-4)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad, 1e-4, 1e-4)
