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
import mindspore as ms
import mindspore.nn as nn
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn
from tests.mark_utils import arg_mark


class Net1(nn.Cell):
    def construct(self, x):
        output = x.hardshrink()
        return output


class Net2(nn.Cell):
    def construct(self, x, lambd=0.5):
        output = x.hardshrink(lambd)
        return output


def hardshrink_expect_forward_func(x, lambd=0.5):
    result = np.zeros_like(x, dtype=x.dtype)
    for index, _ in np.ndenumerate(x):
        if x[index] > lambd or x[index] < (-1 * lambd):
            result[index] = x[index]
        else:
            result[index] = 0
    return result


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
    'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_hardshrink(mode):
    """
    Feature: tensor.hardshrink
    Description: Verify the result of tensor.hardshrink
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net1()
    x = generate_numpy_ndarray_by_randn((3, 4, 5, 6), np.float32, 'x')
    output_1 = net(ms.Tensor(x))
    expect_1 = hardshrink_expect_forward_func(x)
    net2 = Net2()
    output_2 = net2(ms.Tensor(x), 5)
    expect_2 = hardshrink_expect_forward_func(x, 5)
    assert np.allclose(output_1.asnumpy(), expect_1, rtol=1e-3)
    assert np.allclose(output_2.asnumpy(), expect_2, rtol=1e-3)
