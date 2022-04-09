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


class LerpNet(nn.Cell):
    def __init__(self):
        super(LerpNet, self).__init__()
        self.lerp = P.Lerp()

    def construct(self, start, end, weight):
        output = self.lerp(start, end, weight)
        return output


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
