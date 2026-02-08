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
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, mat, vec, beta=1, alpha=1):
    return beta * x + alpha * (mat @ vec)

class Net(nn.Cell):
    def construct(self, x, mat, vec, beta=1, alpha=1):
        return x.addmv(mat, vec, beta=beta, alpha=alpha)

@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_addmv(mode):
    """
    Feature: tensor.addmv
    Description: Verify the result of tensor.addmv
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x_np = generate_random_input((6,), np.float32)
    x = Tensor(x_np, ms.float32)
    mat_np = generate_random_input((6, 3), np.float32)
    mat = Tensor(mat_np, ms.float32)
    vec_np = generate_random_input((3,), np.float32)
    vec = Tensor(vec_np, ms.float32)
    beta = 1.0
    alpha = 1.0
    output = net(x, mat, vec, beta=beta, alpha=alpha)
    expected = generate_expect_forward_output(x_np, mat_np, vec_np, beta=beta, alpha=alpha)
    assert np.allclose(output.asnumpy(), expected)
