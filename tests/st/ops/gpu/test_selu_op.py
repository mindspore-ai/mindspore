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
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context


class SeluOpNet(nn.Cell):
    def __init__(self):
        super(SeluOpNet, self).__init__()
        self.selu = P.SeLU()

    def construct(self, input_x):
        output = self.selu(input_x)
        return output


def selu_op_np_bencmark(input_x):
    """
    Feature: generate a selu numpy benchmark.
    Description: The input shape need to match to output shape.
    Expectation: match to np mindspore SeLU.
    """
    alpha = 1.67326324
    scale = 1.05070098
    alpha_dot_scale = scale * alpha
    result = np.zeros_like(input_x, dtype=input_x.dtype)
    for index, _ in np.ndenumerate(input_x):
        if input_x[index] >= 0.0:
            result[index] = scale * input_x[index]
        else:
            result[index] = alpha_dot_scale * np.expm1(input_x[index])
    return result


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
@pytest.mark.parametrize("data_shape", [(4,), (3, 4), (4, 5, 7)])
def test_selu_op(data_type, data_shape):
    """
    Feature: Test Selu.
    Description: The input shape need to match to output shape.
    Expectation: match to np benchmark.
    """
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    input_data = np.random.random(data_shape).astype(data_type)
    benchmark_output = selu_op_np_bencmark(input_data)
    context.set_context(mode=context.GRAPH_MODE)
    selu = SeluOpNet()
    output = selu(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = selu(Tensor(input_data))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
