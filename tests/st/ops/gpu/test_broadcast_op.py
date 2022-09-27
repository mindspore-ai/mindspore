# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np = np.random.rand(10, 20).astype(np.float32)
    x2_np = np.random.rand(10, 20).astype(np.float32)
    x1_np_int32 = np.random.randint(0, 100, (10, 20)).astype(np.int32)
    x2_np_int32 = np.random.randint(0, 100, (10, 20)).astype(np.int32)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)
    output_ms = P.Greater()(Tensor(x1_np_int32), Tensor(x2_np_int32))
    output_np = x1_np_int32 > x2_np_int32
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)
    output_ms = P.Less()(Tensor(x1_np_int32), Tensor(x2_np_int32))
    output_np = x1_np_int32 < x2_np_int32
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.RealDiv()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mul()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np * x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Sub()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np - x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np)
    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)

    output_ms = P.Mod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.fmod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.FloorMod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.mod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Atan2()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.arctan2(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np = np.random.rand(10, 20).astype(np.float16)
    x2_np = np.random.rand(10, 20).astype(np.float16)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.RealDiv()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mul()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np * x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Sub()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np - x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np)
    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)

    output_ms = P.Mod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.fmod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.FloorMod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.mod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Atan2()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.arctan2(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    x2_np = np.random.rand(1, 4, 1, 6).astype(np.float32)
    x1_np_int32 = np.random.randint(0, 100, (3, 1, 5, 1)).astype(np.int32)
    x2_np_int32 = np.random.randint(0, 100, (3, 1, 5, 1)).astype(np.int32)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)
    output_ms = P.Greater()(Tensor(x1_np_int32), Tensor(x2_np_int32))
    output_np = x1_np_int32 > x2_np_int32
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)
    output_ms = P.Less()(Tensor(x1_np_int32), Tensor(x2_np_int32))
    output_np = x1_np_int32 < x2_np_int32
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.RealDiv()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mul()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np * x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Sub()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np - x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np)
    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)

    output_ms = P.Mod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.fmod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.FloorMod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.mod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Atan2()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.arctan2(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_diff_dims():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np = np.random.rand(2).astype(np.float32)
    x2_np = np.random.rand(2, 1).astype(np.float32)
    x1_np_int32 = np.random.randint(0, 100, (2)).astype(np.int32)
    x2_np_int32 = np.random.randint(0, 100, (2, 1)).astype(np.int32)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    output_ms = P.Greater()(Tensor(x1_np_int32), Tensor(x2_np_int32))
    output_np = x1_np_int32 > x2_np_int32
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)
    output_ms = P.Less()(Tensor(x1_np_int32), Tensor(x2_np_int32))
    output_np = x1_np_int32 < x2_np_int32
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.RealDiv()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mul()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np * x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Sub()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np - x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np)
    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)

    output_ms = P.Mod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.fmod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.FloorMod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.mod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Atan2()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.arctan2(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_diff_dims_float64():
    """
    Feature: ALL To ALL
    Description: test cases for broadcast operations execpted for DivNoNan
    Expectation: the result match numpy results
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np = np.random.rand(2).astype(np.float32)
    x2_np = np.random.rand(2, 1).astype(np.float32)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.RealDiv()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mul()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np * x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Sub()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np - x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.fmod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.FloorMod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.mod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Atan2()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.arctan2(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float16)
    x2_np = np.random.rand(1, 4, 1, 6).astype(np.float16)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.RealDiv()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Mul()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np * x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Sub()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np - x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np / x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np)
    output_ms = P.DivNoNan()(Tensor(x1_np), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)

    output_ms = P.Mod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.fmod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.FloorMod()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.mod(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Atan2()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.arctan2(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_divnonan_int8():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np_int8 = np.random.randint(1, 100, (10, 20)).astype(np.int8)
    x2_np_int8 = np.random.randint(1, 100, (10, 20)).astype(np.int8)

    output_ms = P.DivNoNan()(Tensor(x1_np_int8), Tensor(x2_np_int8))
    output_np = x1_np_int8 // x2_np_int8
    print(output_ms.asnumpy(), output_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np_int8)
    output_ms = P.DivNoNan()(Tensor(x1_np_int8), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_divnonan_uint8():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    np.random.seed(42)
    x1_np_uint8 = np.random.randint(1, 100, (10, 20)).astype(np.uint8)
    x2_np_uint8 = np.random.randint(1, 100, (10, 20)).astype(np.uint8)

    output_ms = P.DivNoNan()(Tensor(x1_np_uint8), Tensor(x2_np_uint8))
    output_np = x1_np_uint8 // x2_np_uint8
    print(output_ms.asnumpy(), output_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    x2_np_zero = np.zeros_like(x2_np_uint8)
    output_ms = P.DivNoNan()(Tensor(x1_np_uint8), Tensor(x2_np_zero))
    assert np.allclose(output_ms.asnumpy(), x2_np_zero)
