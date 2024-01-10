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
import mindspore as ms


@test_utils.run_with_cell
def conj_forward_func(x):
    return ops.auto_generate.conj(x)


@test_utils.run_with_cell
def conj_backward_func(x):
    return ops.grad(conj_forward_func, (0,))(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_conj_forward(mode):
    """
    Feature: Ops.
    Description: test op conj.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array((1.3 + 0.4j, -1.5 - 0.2j, 0 + 0.2j), np.complex64)
    input_x = ms.Tensor(x, ms.complex64)
    output = conj_forward_func(input_x)
    expect = np.array((1.3 - 0.4j, -1.5 + 0.2j, 0 - 0.2j), np.complex64)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_conj_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op conj.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array((1.3 + 0.4j, -1.5 - 0.2j, 0 + 0.2j), np.complex64)
    input_x = ms.Tensor(x, ms.complex64)
    grads = conj_backward_func(input_x)
    expect = np.array((1. + 0.j, 1. - 0.j, 1. + 0.j), np.complex64)
    assert np.allclose(grads.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_conj_vmap(mode):
    """
    Feature: test vmap function.
    Description: test conj op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0)
    np_x = np.random.randn(1, 1, 6, 6, 3, 6).astype(np.complex64)
    x = ms.Tensor(np_x)
    expect = np.conj(np_x).astype(np.complex64)
    nest_vmap = ops.vmap(ops.vmap(conj_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    vmap_out = nest_vmap(x)
    assert np.allclose(vmap_out.asnumpy(), expect)
