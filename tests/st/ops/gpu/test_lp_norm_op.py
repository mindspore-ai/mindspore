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
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from mindspore.common import dtype as ms_type


class LpNormNet(nn.Cell):
    def __init__(self, axis, p=2, keep_dims=False, epsilon=1e-12):
        super(LpNormNet, self).__init__()
        self.lp_norm = P.LpNorm(axis, p, keep_dims, epsilon)

    def construct(self, input_x):
        output = self.lp_norm(input_x)
        return output


class LpNormVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(LpNormVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x)


def lp_norm_np_bencmark(data_type):
    """
    Feature: generate a LpNorm numpy benchmark.
    Description: The input shape need to match input shape.
    Expectation: match to np mindspore LpNorm.
    """
    result = np.array([9.165152, 10.954452]).astype(data_type)
    return result


def lp_norm_vmap_case(data_type):
    """
    Feature: test lp_norm vamp feature.
    Description: test special case.
    Expectation: match to mindspore.ops.LpNorm.
    """
    # Case : in_axes input_x batch remains 0
    input_x = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]).astype(data_type)
    in_axes = 0
    out_axes = 0
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    benchmark_output = np.array([[9.165152, 10.954452], [9.165152, 10.954452], [9.165152, 10.954452]]).astype(data_type)
    axis = [0, 1]
    p = 2
    keep_dims = False
    lp_norm = LpNormNet(axis, p, keep_dims)
    output = LpNormVMapNet(lp_norm, in_axes, out_axes)(Tensor(input_x))
    assert np.allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_lp_norm_op(data_type):
    """
    Feature: Test LpNorm.
    Description: The input shape need match to output shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(data_type)
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    benchmark_output = lp_norm_np_bencmark(data_type)
    axis = [0, 1]
    p = 2
    keep_dims = False
    lp_norm = LpNormNet(axis, p, keep_dims)
    output = lp_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = lp_norm(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_lp_norm_vmap_gpu(data_type):
    """
    Feature: test LpNorm vmap on GPU.
    Description: inputs(input_x) with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    lp_norm_vmap_case(data_type)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_lp_norm_dy_shape(data_type):
    """
    Feature: Test LpNorm DyNamicShape.
    Description: The input data type only float16 and float32.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    axis = [0, 1]
    p = 2
    keep_dims = False
    lp_norm_net = LpNormNet(axis, p, keep_dims)
    input_x_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(data_type)
    loss = 1e-6
    ms_data_type = ms_type.float32
    if data_type == np.float16:
        ms_data_type = ms_type.float16
        loss = 1e-3
    benchmark_output = lp_norm_np_bencmark(data_type)
    input_dyn = Tensor(shape=[2, 2, None], dtype=ms_data_type)
    lp_norm_net.set_inputs(input_dyn)
    ms_result = lp_norm_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = lp_norm_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
