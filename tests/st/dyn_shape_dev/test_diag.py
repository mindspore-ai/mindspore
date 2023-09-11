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

from mindspore import ops, context
import mindspore as ms



@ms.jit
def diag_forward_func(input_x):
    return ops.auto_generate.diag(input_x)


@ms.jit
def diag_backward_func(input_x):
    return ops.grad(diag_forward_func, (0,))(input_x)


@ms.jit
def diag_vmap_func(x):
    return ops.vmap(diag_forward_func, in_axes=0, out_axes=0)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
def test_diag_forward():
    """
    Feature: Ops.
    Description: test op diag.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = ms.Tensor(np.array([1, 2, 5]), ms.float16)
    out = diag_forward_func(input_x)
    print("out:", out)
    expect = np.array([[1, 0, 0],
                       [0, 2, 0],
                       [0, 0, 5]]).astype(np.float16)
    assert (out.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
def test_diag_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op diag.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = ms.Tensor([[1.3, 2.5, -2.1], [-4.7, 2.5, 1.0]], ms.float32)
    out = diag_backward_func(input_x)
    expect = np.array([[1, 1, 1],
                       [1, 1, 1]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.env_onecard
# @pytest.mark.platform_x86_cpu CPU不支持batch_rank, 进而不支持vmap
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
def test_diag_vmap():
    """
    Feature: test vmap function.
    Description: test diag op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = ms.Tensor([[3, 1, 4], [1, 5, 9]], ms.float32)
    out = diag_vmap_func(input_x)
    expect = np.array([[[3, 0, 0], [0, 1, 0], [0, 0, 4]], [[1, 0, 0], [0, 5, 0], [0, 0, 9]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()
