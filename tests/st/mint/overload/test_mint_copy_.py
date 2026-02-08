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
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y):
        return x.copy_(y)


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_copy_(mode):
    """
    Feature: test Tensor.copy_
    Description: Verify the result of Tensor.copy_
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = generate_random_input((2, 2, 3, 4), np.float32)
    y = generate_random_input((2, 2, 3, 4), np.float32)
    z = generate_random_input((2, 1, 4), np.float32)  # broadcast

    expect_z = np.expand_dims(z.repeat(3, axis=1), axis=0).repeat(2, axis=0)

    net = Net()
    y_output = net(Tensor(x), Tensor(y))
    z_output = net(Tensor(x), Tensor(z))
    assert np.allclose(y_output.asnumpy(), y, rtol=1e-5, equal_nan=True)
    assert np.allclose(z_output.asnumpy(), expect_z, rtol=1e-5, equal_nan=True)
