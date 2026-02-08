# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x, tensor1, tensor2, value):
        output = x.addcdiv(tensor1, tensor2, value=value)
        return output

def generate_random_input(shape1, shape2, shape3, dtype):
    x1 = ms.Tensor(np.random.normal(0, 10, size=shape1).astype(dtype))
    x2 = ms.Tensor(np.random.normal(0, 10, size=shape2).astype(dtype))
    x3 = ms.Tensor(np.random.normal(0, 10, size=shape3).astype(dtype))
    return x1, x2, x3

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_addcdiv(mode):
    """
    Feature: tensor.flatten
    Description: Verify the result of flatten in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x, tensor1, tensor2 = generate_random_input([1, 3], [3, 1], [1, 3], np.float32)
    output = net(x, tensor1, tensor2, value=2)
    expect_output = x + tensor1 / tensor2 * 2
    assert np.allclose(output.asnumpy(), expect_output)
