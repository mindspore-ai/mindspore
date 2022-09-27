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
from mindspore.ops import operations as P


def hshrink_op_np_bencmark(input_x, lambd):
    """
    Feature: generate a hshrink numpy benchmark.
    Description: The input shape need to match to output shape.
    Expectation: match to nn mindspore HShrink.
    """
    result = np.zeros_like(input_x, dtype=input_x.dtype)
    for index, _ in np.ndenumerate(input_x):
        if input_x[index] > lambd or input_x[index] < (-1 * lambd):
            result[index] = input_x[index]
        else:
            result[index] = 0
    return result


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
@pytest.mark.parametrize("lambd", [0.5])
def test_hshrink(dtype, data_shape, lambd):
    """
    Feature: HShrink gpu kernel
    Description: test the rightness of HShrink gpu kernel
    Expectation: the output is same as hshrink_op_np_bencmark output
    """
    class NetHShrink(nn.Cell):
        def __init__(self):
            super(NetHShrink, self).__init__()
            self.hard_shrink = P.HShrink(lambd)

        def construct(self, input_x):
            return self.hard_shrink(input_x)

    input_data = np.random.uniform(
        low=-1, high=1, size=data_shape).astype(dtype)
    benchmark_output = hshrink_op_np_bencmark(input_data, lambd)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    hshrink = NetHShrink()
    output = hshrink(Tensor(input_data))
    assert np.allclose(output.asnumpy(), benchmark_output)
