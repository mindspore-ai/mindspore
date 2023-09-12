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


@ms.jit
def cos_forward_func(x):
    return ops.auto_generate.cos(x)


@ms.jit
def cos_backward_func(x):
    return ops.grad(cos_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_cos_forward():
    """
    Feature: Ops.
    Description: test op cos.
    Expectation: expect correct result.
    """
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    output = cos_forward_func(input_x)
    expect = np.cos(x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_cos_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op cos.
    Expectation: expect correct result.
    """
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    grads = cos_backward_func(input_x)
    expect = np.array([-0.8912074, -0.5984722, 0.997495]).astype(np.float32)
    assert np.allclose(grads.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_cos_vmap():
    """
    Feature: test vmap function.
    Description: test cos op vmap.
    Expectation: expect correct result.
    """
    in_axes = (0)
    np_x = np.array([[[1.1, 0.9], [2.2, 1.8]], [[4.6, 1.3], [2.4, 2.6]],
                     [[1.0, 1.0], [2.0, 2.7]], [[1.3, 1.7], [2.9, 2.8]],
                     [[1.1, 1.4], [2.6, 2.0]], [[1.2, 1.4], [2.0, 2.4]],
                     [[1.5, 1.4], [2.3, 2.0]], [[1.8, 1.0], [2.9, 2.0]]]).astype(np.float32)
    x = ms.Tensor(np_x)
    expect = np.cos(np_x)
    nest_vmap = ops.vmap(ops.vmap(cos_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    vmap_out = nest_vmap(x)
    assert np.allclose(vmap_out.asnumpy(), expect)
