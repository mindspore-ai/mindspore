# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore import ops, Tensor, dtype, jit


def test_cast():
    """
    Feature: test cast operator
    Description: Cast original data type to target data type
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    type_dst = ms.float32
    cast = ops.Cast()
    result = cast(input_x, type_dst)
    assert result.dtype == type_dst


@jit
def expand_tensor(a, b):
    out = ops.tile(a, b)
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile_eliminate():
    """
    Feature: tile_eliminate
    Description: All value of multiplier is '1' but length of multiplier is greater than tensor dims, can't do eliminate
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor_ = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    out = ops.tile(tensor_, (1, 1, 1))
    assert out.shape == (1, 448, 448)
    out = ops.tile(tensor_, (1, 1, 1, 1))
    assert out.shape == (1, 1, 448, 448)
    out = expand_tensor(tensor_, (1, 1, 1))
    assert out.shape == (1, 448, 448)
    out = expand_tensor(tensor_, (1, 1, 1, 1))
    assert out.shape == (1, 1, 448, 448)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_shape_raise():
    """
    Feature: shape raise.
    Description: Test raise.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor0 = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    tensor1 = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    with pytest.raises(TypeError):
        ops.shape([tensor0, tensor1])
