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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetHShrinkGrad(nn.Cell):
    def __init__(self, lambd=0.5):
        super(NetHShrinkGrad, self).__init__()
        self.hard_shrink_grad = G.HShrinkGrad(lambd)

    def construct(self, grad, input_x):
        return self.hard_shrink_grad(grad, input_x)


def hshrink_grad_op_np_bencmark(grad, input_x, lambd):
    """
    Feature: generate a hshrink grad numpy benchmark.
    Description: The input shape need to match to output shape.
    Expectation: match to mindspore HShrinkGrad.
    """
    result = np.zeros_like(grad, dtype=grad.dtype)
    for index, _ in np.ndenumerate(grad):
        if input_x[index] > lambd or input_x[index] < (-1 * lambd):
            result[index] = grad[index]
        else:
            result[index] = 0
    return result


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
@pytest.mark.parametrize("lambd", [0.5])
def test_hshrink_grad(dtype, data_shape, lambd):
    """
    Feature: HShrinkGrad cpu kernel
    Description: test the rightness of HShrinkGrad cpu kernel
    Expectation: the output is same as hshrink_grad_op_np_bencmark output
    """
    grad_data = np.random.random(data_shape).astype(dtype)
    input_data = np.random.uniform(
        low=-1, high=1, size=data_shape).astype(dtype)
    benchmark_output = hshrink_grad_op_np_bencmark(
        grad_data, input_data, lambd)
    hshrink_grad = NetHShrinkGrad(lambd)
    output = hshrink_grad(Tensor(grad_data), Tensor(input_data))
    assert np.allclose(output.asnumpy(), benchmark_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('axis', [2])
def test_hshrink_grad_vmap_cpu(axis):
    """
    Feature: HShrinkGrad cpu kernel
    Description: test the rightness of HShrinkGrad cpu kernel vmap feature.
    Expectation: Success.
    """
    hshrink = NetHShrinkGrad()

    def hshrink_func(grad, input_x):
        """hshrink_func"""
        return hshrink(grad, input_x)

    grad_data = np.random.random((3, axis, 5)).astype(np.float32)
    input_data = np.random.uniform(
        low=-1, high=1, size=(3, axis, 5)).astype(np.float32)

    output_vmap = vmap(hshrink_func, in_axes=(1, 1))(
        Tensor(grad_data), Tensor(input_data))

    output_manually = hshrink(Tensor(
        grad_data.transpose(1, 0, 2)), Tensor(input_data.transpose(1, 0, 2)))

    assert np.array_equal(output_vmap.asnumpy(), output_manually)
