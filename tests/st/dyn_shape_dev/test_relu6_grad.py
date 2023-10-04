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
import mindspore as ms
from mindspore import ops
from mindspore.ops.auto_generate.gen_ops_def import _relu6_grad
import test_utils


@test_utils.run_with_cell
def relu6_grad_func(dy, x):
    return _relu6_grad(dy, x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_relu6_grad(mode):
    """
    Feature: Ops.
    Description: test op relu6 grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy = ms.Tensor(np.array([[[[1, 0, 1],
                               [0, 1, 0],
                               [1, 1, 1]]]]).astype(np.float32))
    x = ms.Tensor(np.array([[[[-1, 1, 1],
                              [1, -1, 1],
                              [1, 1, -1]]]]).astype(np.float32))
    out = relu6_grad_func(dy, x)
    expect_out = np.array([[[[0, 0, 1],
                             [0, 0, 0],
                             [1, 1, 0]]]]).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_relu6_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test relu6 op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = (-1, -1)
    dy = ms.Tensor(np.random.uniform(size=(4, 3, 2)).astype(np.float32))
    x = ms.Tensor(np.random.uniform(low=-5, high=10, size=(4, 3, 2)).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(relu6_grad_func, in_axes=axes, out_axes=-1), in_axes=axes, out_axes=-1)
    out = net_vmap(dy, x)
    expect_out = relu6_grad_func(dy, x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
