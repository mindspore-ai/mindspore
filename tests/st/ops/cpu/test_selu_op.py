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
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from mindspore.common import dtype as ms_type


class SeluNet(nn.Cell):
    def __init__(self):
        super(SeluNet, self).__init__()
        self.selu = P.SeLU()

    def construct(self, input_x):
        output = self.selu(input_x)
        return output


class SeluVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(SeluVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x)


def selu_np_bencmark(input_x):
    """
    Feature: generate a selu numpy benchmark.
    Description: The input shape need to match to output shape.
    Expectation: match to np mindspore SeLU.
    """
    alpha = 1.67326324
    scale = 1.05070098
    result = np.zeros_like(input_x, dtype=input_x.dtype)

    for index, _ in np.ndenumerate(input_x):
        if input_x[index] >= 0.0:
            result[index] = scale * input_x[index]
        else:
            result[index] = scale * alpha * np.expm1(input_x[index])
    return result


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_shape", [(4,), (3, 4), (4, 5, 7)])
@pytest.mark.parametrize("data_type", [np.int8, np.int32, np.float32, np.float16])
def test_selu(data_shape, data_type):
    """
    Feature: Test Selu.
    Description: The input shape need to match to output shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_data = np.random.random(data_shape).astype(data_type)
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    benchmark_output = selu_np_bencmark(input_data)
    selu = SeluNet()
    output = selu(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = selu(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_selu_vmap_cpu():
    """
    Feature: test SeLU vmap on CPU.
    Description: inputs(input_x) with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    data_type = np.float32
    data_shape = (10, 5, 7)
    input_data = np.random.random(data_shape).astype(data_type)
    in_axes = 0
    out_axes = 0
    error = 1e-6
    benchmark_output = selu_np_bencmark(input_data)
    selu = SeluNet()
    output = SeluVMapNet(selu, in_axes, out_axes)(Tensor(input_data))
    assert np.allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_selu_dy_shape():
    """
    Feature: Test SeLU DynamicShape.
    Description: The input data type only float16 and float32.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data_type = np.float32
    ms_data_type = ms_type.float32
    data_shape = (10, 5, 7)
    input_x_np = np.random.random(data_shape).astype(data_type)
    loss = 1e-6
    benchmark_output = selu_np_bencmark(input_x_np)
    selu_net = SeluNet()
    input_dyn = Tensor(shape=[10, 5, None], dtype=ms_data_type)
    selu_net.set_inputs(input_dyn)
    ms_result = selu_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = selu_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
