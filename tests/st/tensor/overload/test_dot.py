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
import pytest
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn


class DotNet(nn.Cell):
    def construct(self, x, y):
        return x.dot(y)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return np.sum(x * y.transpose())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_dot(mode):
    """
    Feature: test Tensor.dot
    Description: Verify the result of Tensor.dot
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = generate_random_input((10,), np.float32)
    y = generate_random_input((10,), np.float32)
    net = DotNet()
    output = net(ms.Tensor(x, dtype=ms.float32), ms.Tensor(y, dtype=ms.float32))
    expect_output = generate_expect_forward_output(x, y)
    assert np.allclose(output.asnumpy(), expect_output, rtol=1e-4, equal_nan=True)
