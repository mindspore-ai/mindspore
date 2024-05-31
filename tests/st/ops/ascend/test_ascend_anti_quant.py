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

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor
from mindspore.ops.operations._inner_ops import AntiQuant


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, scale, sqrt_mode=False, offset=None):
    if offset is None:
        offset = np.zeros_like(scale)
    if sqrt_mode:
        return scale * scale * (x + offset).astype(np.float16)
    return scale * (x + offset).astype(np.float16)

def antiquant_forward_func(x, scale, sqrt_mode=False, offset=None):
    return AntiQuant(sqrt_mode)(x, scale, offset)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('sqrt_mode', [True, False])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_antiquant_offset_none(sqrt_mode, mode):
    """
    Feature: pyboost antiquant.
    Description: test function antiquant when offset is None.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.int8)
    scale = generate_random_input((5,), np.float32)
    output = antiquant_forward_func(Tensor(x), Tensor(scale), sqrt_mode)
    expect = generate_expect_forward_output(x, scale, sqrt_mode)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('sqrt_mode', [True, False])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_antiquant_offset_not_none(sqrt_mode, mode):
    """
    Feature: pyboost function.
    Description: test function antiquant when offset is not None.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.int8)
    scale = generate_random_input((5,), np.float32)
    offset = generate_random_input((5,), np.float32)
    output = antiquant_forward_func(Tensor(x), Tensor(scale), sqrt_mode, Tensor(offset))
    expect = generate_expect_forward_output(x, scale, sqrt_mode, offset)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
