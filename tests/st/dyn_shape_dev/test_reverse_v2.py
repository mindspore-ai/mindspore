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
from mindspore import ops
import mindspore as ms
import test_utils


@test_utils.run_with_cell
def reverse_v2_forward_func(x):
    return ops.auto_generate.reverse_v2(x, axis=(0,))


@test_utils.run_with_cell
def reverse_v2_backward_func(x):
    return ops.grad(reverse_v2_forward_func, (0,))(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_reverse_v2(mode):
    """
    Feature: Ops.
    Description: test op reverse_v2.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32))
    out = reverse_v2_forward_func(x)
    expect_out = np.flip(x.asnumpy(), (0,))
    assert (out.asnumpy() == expect_out).all()

    grad = reverse_v2_backward_func(x)
    expect_grad = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1]]).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
#@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="Probability of failure.")
def test_reverse_v2_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reverse_v2 op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = -1
    x = ms.Tensor(np.array([[[0, 1, 2, 3],
                             [4, 5, 6, 7],
                             [8, 9, 10, 11]],

                            [[12, 13, 14, 15],
                             [16, 17, 18, 19],
                             [20, 21, 22, 23]]]).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(reverse_v2_forward_func, in_axes=axes, out_axes=axes), in_axes=axes, out_axes=axes)
    out = net_vmap(x)
    expect_out = np.array([[[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]],

                           [[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11]]]).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()
