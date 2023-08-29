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
from mindspore import ops, Tensor


@ms.jit
def gather_forward_func(input_params, input_indices, axis, batch_dims=0):
    return ops.auto_generate.gather(input_params, input_indices, axis, batch_dims)


@ms.jit
def gather_backward_func(input_params, input_indices, axis, batch_dims=0):
    return ops.grad(gather_forward_func, (0,))(input_params, input_indices, axis, batch_dims)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_gather_forward():
    """
    Feature: Ops.
    Description: test op gather.
    Expectation: expect correct result.
    """
    input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), ms.float32)
    input_indices = Tensor(np.array([0, 2, 1]), ms.int32)
    axis = 1
    batch_dims = 1
    output = gather_forward_func(input_params, input_indices, axis, batch_dims)
    expect = [1., 7., 10.]
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_gather_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op gather.
    Expectation: expect correct result.
    """
    input_params = Tensor(np.array([1, 2, 3]), ms.float32)
    input_indices = Tensor(np.array([0, 2, 1]), ms.int32)
    axis = 0
    output = gather_backward_func(input_params, input_indices, axis)
    expect = [1., 1., 1.]
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_gather_vmap():
    """
    Feature: test vmap function.
    Description: test gather op vmap.
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [1, 2]]).astype(np.int32))
    axis = 0
    nest_vmap = ops.vmap(gather_forward_func, in_axes=(0, 0, None), out_axes=0)
    output = nest_vmap(x, indices, axis)
    expect = np.array([[1, 2], [5, 6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)
