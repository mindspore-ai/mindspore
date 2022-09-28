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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
@pytest.mark.parametrize("lambd", [0.5])
def test_hshrink_grad(dtype, data_shape, lambd):
    """
    Feature: HShrinkGrad gpu kernel
    Description: test the rightness of HShrinkGrad gpu kernel
    Expectation: the output is same as hshrink_grad_op_np_bencmark output
    """
    class NetHShrinkGrad(nn.Cell):
        def __init__(self):
            super(NetHShrinkGrad, self).__init__()
            self.hard_shrink_grad = G.HShrinkGrad(lambd)

        def construct(self, grad, input_x):
            return self.hard_shrink_grad(grad, input_x)

    grad_data = np.random.random(data_shape).astype(dtype)
    input_data = np.random.uniform(
        low=-1, high=1, size=data_shape).astype(dtype)
    benchmark_output = hshrink_grad_op_np_bencmark(
        grad_data, input_data, lambd)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    hshrink_grad = NetHShrinkGrad()
    output = hshrink_grad(Tensor(grad_data), Tensor(input_data))
    assert np.allclose(output.asnumpy(), benchmark_output)
