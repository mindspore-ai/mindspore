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


def soft_shrink_op_np_bencmark(input_x, lambd):
    result = input_x.asnumpy().copy()
    size = input_x.size
    result = result.reshape(size)

    for index in range(size):
        if result[index] > lambd:
            result[index] = result[index] - lambd
        elif result[index] < -lambd:
            result[index] = result[index] + lambd
        else:
            result[index] = 0

    result = result.reshape(input_x.shape)
    return result


class SoftShrinkNet(nn.Cell):
    def __init__(self, lambd):
        super(SoftShrinkNet, self).__init__()
        self.soft_shrink = P.SoftShrink(lambd)

    def construct(self, input_x):
        return self.soft_shrink(input_x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
@pytest.mark.parametrize("lambd", [0.5])
def test_soft_shrink(dtype, data_shape, lambd):
    """
    Feature: SoftShrink cpu kernel
    Description: test the rightness of SoftShrink cpu kernel
    Expectation: the output is same as soft_shrink_op_np_bencmark output
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    data = np.random.uniform(low=-1, high=1, size=data_shape).astype(dtype)
    input_tensor = Tensor(data)
    benchmark_output = soft_shrink_op_np_bencmark(input_tensor, lambd)

    soft_shrink_net = SoftShrinkNet(lambd)
    output = soft_shrink_net(input_tensor)
    np.testing.assert_array_almost_equal(output.asnumpy(), benchmark_output)
