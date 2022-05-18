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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops.functional import vmap
from mindspore.ops.operations.math_ops import Lerp


class LerpNet(nn.Cell):
    def __init__(self):
        super(LerpNet, self).__init__()
        self.lerp = Lerp()

    def construct(self, start, end, weight):
        output = self.lerp(start, end, weight)
        return output


class LerpVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(LerpVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, start, end, weight):
        return vmap(self.net, self.in_axes, self.out_axes)(start, end, weight)


def lerp_vmap_case():
    """
    Feature: test lerp vamp feature.
    Description: test special case.
    Expectation: match to mindspore.ops.Lerp.
    """
    # Case 1: in_axes start batch remains 0,  other remains None.
    start = Tensor(np.array([[1., 2., 3., 4.], [1., 3., 5., 7.]]).astype(np.float32))
    end = Tensor(np.array([10., 10., 10., 10.]).astype(np.float32))
    weight = 0.5
    benchmark_output = np.array([[5.5, 6., 6.5, 7.], [5.5, 6.5, 7.5, 8.5]]).astype(np.float32)
    in_axes = (0, None, None)
    out_axes = 0
    output = LerpVMapNet(LerpNet(), in_axes, out_axes)(start, end, weight)
    assert np.allclose(output.asnumpy(), benchmark_output)

    # Case 2: start remains 0 batch, end remains 1 batch, weight remains None.
    start = Tensor(np.array([[1., 2., 3., 4.], [1., 3., 5., 7.]]).astype(np.float32))
    end = Tensor(np.array([[10., 4.], [10., 4.], [10., 4.], [10., 4.]]).astype(np.float32))
    weight = 0.5
    benchmark_output = np.array([[5.5, 6., 6.5, 7.], [2.5, 3.5, 4.5, 5.5]]).astype(np.float32)
    in_axes = (0, 1, None)
    out_axes = 0
    output = LerpVMapNet(LerpNet(), in_axes, out_axes)(start, end, weight)
    assert np.allclose(output.asnumpy(), benchmark_output)

    # Case 3: start remains 1 batch,end remains 1 batch, weight remains 0 batch.
    start = Tensor(np.array([[1., 1], [2., 3.], [3., 5.], [4., 7.]]).astype(np.float32))
    end = Tensor(np.array([[10., 4.], [10., 4.], [10., 4.], [10., 4.]]).astype(np.float32))
    weight = Tensor(np.array([0.5, 0.4]).astype(np.float32))
    benchmark_output = np.array([[5.5, 6., 6.5, 7.], [2.2, 3.4, 4.6, 5.8]]).astype(np.float32)
    in_axes = (1, 1, 0)
    out_axes = 0
    output = LerpVMapNet(LerpNet(), in_axes, out_axes)(start, end, weight)
    assert np.allclose(output.asnumpy(), benchmark_output)

    # Case 4: start remain None, end remains 1 batch, weight remains 0 batch.
    start = Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    end = Tensor(np.array([[10., 4.], [10., 4.], [10., 4.], [10., 4.]]).astype(np.float32))
    weight = Tensor(np.array([0.5, 0.4]).astype(np.float32))
    benchmark_output = np.array([[5.5, 6., 6.5, 7.], [2.2, 2.8, 3.4, 4.]]).astype(np.float32)
    in_axes = (None, 1, 0)
    out_axes = 0
    output = LerpVMapNet(LerpNet(), in_axes, out_axes)(start, end, weight)
    assert np.allclose(output.asnumpy(), benchmark_output)


def lerp_np_bencmark(start, end, weight):
    """
    Feature: generate a lerp numpy benchmark.
    Description: The input shape may need to broadcast.
    Expectation: match to np mindspore lerp.
    """
    end = np.broadcast_to(end, start.shape)
    weight = np.broadcast_to(weight, start.shape)
    result = start + weight * (end - start)
    return result


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("data_shape", [(4,), (3, 4), (4, 5, 7)])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_lerp(data_shape, data_type):
    """
    Feature: Test Lerp.
    Description: The input shape may need to broadcast.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    start = np.random.random(data_shape).astype(data_type)
    end = np.ones(data_shape).astype(data_type)
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    weight = 0.5
    benchmark_output = lerp_np_bencmark(start, end, weight)
    lerp = LerpNet()
    output = lerp(Tensor(start), Tensor(end), Tensor(np.array(weight, dtype=data_type)))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = lerp(Tensor(start), Tensor(end), Tensor(np.array(weight, dtype=data_type)))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lerp_vmap_cpu():
    """
    Feature: test Lerp vmap on CPU.
    Description: inputs(start, end, weight) with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    lerp_vmap_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    lerp_vmap_case()
