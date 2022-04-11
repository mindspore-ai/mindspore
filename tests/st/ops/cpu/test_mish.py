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


class MishNet(nn.Cell):
    def __init__(self):
        super(MishNet, self).__init__()
        self.mish = P.Mish()

    def construct(self, x):
        output = self.mish(x)
        return output


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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
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
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    benchmark_output = mish_np_bencmark(x)
    mish = MishNet()
    output = mish(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = mish(Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error)
