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
    def construct(self, x, indices, axis=None, mode='clip'):
        return x.take(indices, axis, mode)


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_expect_output(x, indices):
    reshape_input = x.reshape(6)
    out = []
    for item in indices:
        out.append(reshape_input[item])
    return ms.Tensor(out)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows',
                'cpu_macos', 'platform_gpu', 'platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_take(mode):
    """
    Feature: Tensor.take
    Description: Verify the result of Tensor.take
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"}, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(generate_random_input((2, 3), np.float32))
    indices = ms.Tensor([2], ms.int64)
    net = Net()
    output = net(x, indices)
    expect_output = generate_expect_output(x, indices)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
