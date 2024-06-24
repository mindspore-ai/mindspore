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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as ms_type


class MishNet(nn.Cell):
    def __init__(self):
        super(MishNet, self).__init__()
        self.mish = P.Mish()

    def construct(self, x):
        output = self.mish(x)
        return output


class MishVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(MishVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x)


def mish_np_bencmark(x):
    """
    Feature: generate a mish numpy benchmark.
    Description: The input shape match to input.
    Expectation: match to np mindspore mish.
    """
    result = np.zeros_like(x, dtype=x.dtype)
    for index, _ in np.ndenumerate(x):
        result[index] = x[index] * np.tanh(np.log(np.exp(x[index]) + 1))
    return result


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_shape", [(4,), (3, 4), (4, 5, 7)])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_mish(data_shape, data_type):
    """
    Feature: Test Mish.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.random(data_shape).astype(data_type)
    error = 1e-4
    if data_type == np.float16:
        error = 1e-3
    benchmark_output = mish_np_bencmark(x)
    mish = MishNet()
    output = mish(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = mish(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mish_vmap():
    """
    Feature: Test Mish Vmap on CPU.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data_shape = (10, 4, 5, 7)
    data_type = np.float32
    input_x = np.random.random(data_shape).astype(data_type)
    error = 1e-4
    benchmark_output = mish_np_bencmark(input_x)
    mish = MishNet()
    in_axes = 0
    out_axes = 0
    output = MishVMapNet(mish, in_axes, out_axes)(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mish_dy_shape():
    """
    Feature: Test Mish Dynamic Shape.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    ms_data_type = ms_type.float32
    data_type = np.float32
    data_shape = (4, 5, 7)
    x = np.random.random(data_shape).astype(data_type)
    loss = 1e-4
    benchmark_output = mish_np_bencmark(x)
    mish = MishNet()
    input_dyn = Tensor(shape=[4, 5, None], dtype=ms_data_type)
    mish.set_inputs(input_dyn)
    output = mish(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    input_dyn = Tensor(shape=[4, 5, None], dtype=ms_data_type)
    mish.set_inputs(input_dyn)
    output = mish(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=loss, atol=loss)
