# Copyright 2023 Huawei Technologies Co., Ltd
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
import test_utils

from mindspore import ops
from mindspore import Tensor
import mindspore as ms


@test_utils.run_with_cell
def gelu_grad_forward_func(dy, x, y):
    return ops.auto_generate.gelu_grad(dy, x, y)


@test_utils.run_with_cell
def gelu_grad_backward_func(dy, x, y):
    return ops.grad(gelu_grad_forward_func, (0,))(dy, x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gelu_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op gelu_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy = Tensor(np.array([1.0829641, 1.0860993, 1.0115843]).astype('float32'))
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype('float32'))
    y = Tensor(np.array([1.0, 2.0, 3.0]).astype('float32'))
    out = gelu_grad_forward_func(dy, x, y)
    print("out: ", out)
    expect = np.array([1.1728112, 1.1796116, 1.0233028]).astype('float32')
    assert np.allclose(out.asnumpy(), expect)
