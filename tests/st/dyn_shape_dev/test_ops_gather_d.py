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

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import test_utils


@test_utils.run_with_cell
def gather_d_forward_func(x, dim, index):
    return ops.auto_generate.gather_d(x, dim, index)


@test_utils.run_with_cell
def gather_d_backward_func(x, dim, index):
    return ops.grad(gather_d_forward_func, (0,))(x, dim, index)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_gather_d_forward(mode):
    """
    Feature: Ops.
    Description: test op gather_d.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int32)
    out = gather_d_forward_func(x, dim, index)
    expect = [[1, 1], [4, 3]]
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_gather_d_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op gather_d.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int32)
    grads = gather_d_backward_func(x, dim, index)
    expect = [[2., 0.], [1., 1.]]
    print("grads:", grads)
    assert np.allclose(grads.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_gather_d_vmap(mode):
    """
    Feature: test vmap function.
    Description: test gather_d op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3), ms.float32)
    dim = 1
    index = Tensor(np.zeros([2, 1, 2, 3]).astype(np.int32))
    nest_vmap = ops.vmap(ops.vmap(gather_d_forward_func, in_axes=(-1, None, -1)), in_axes=(-1, None, -1))
    out = nest_vmap(x, dim, index)
    expect = [[[[0.], [6.]], [[3.], [9.]]], [[[1.], [7.]], [[4.], [10.]]], [[[2.], [8.]], [[5.], [11.]]]]
    assert np.allclose(out.asnumpy(), expect)
