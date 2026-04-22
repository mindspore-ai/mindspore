# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Tests for Tensor.clamp.
"""
import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import nn
from mindspore import Tensor


class ClampNet(nn.Cell):
    def construct(self, x, min_value, max_value):
        return x.clamp(min_value, max_value)


class ClipNet(nn.Cell):
    def construct(self, x, min_value, max_value):
        return x.clip(min_value, max_value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clamp(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp with both scalar bounds
                 (ClampScalar path) and tensor bounds (ClampTensor path).
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClampNet()

    # Scalar bounds (ClampScalar path).
    output_ms_case_1 = net(x, 5, 20)
    expect_output_case_1 = np.clip(x_np, 5, 20)
    output_ms_case_2 = net(x, 20, 5)
    expect_output_case_2 = np.clip(x_np, 20, 5)
    assert np.allclose(output_ms_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_ms_case_2.asnumpy(), expect_output_case_2)

    # Tensor bounds (ClampTensor path) — broadcast min/max along axis 0.
    # Covers pyboost ClampTensorCustomizeCall (PYNATIVE_MODE) and the fallback
    # IR builder REG_FALLBACK_BUILDER("ClampTensor") (GRAPH_MODE).
    min_np = np.array([5., 5., 5., 5.], dtype=np.float32)
    max_np = np.array([20., 20., 20., 20.], dtype=np.float32)
    output_tensor = net(x, Tensor(min_np), Tensor(max_np))
    expect_tensor = np.clip(x_np, min_np, max_np)
    assert output_tensor.dtype == ms.float32
    assert np.allclose(output_tensor.asnumpy(), expect_tensor)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clip(mode):
    """
    Feature: test Tensor.clip
    Description: Verify the result of Tensor.clip
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = ClipNet()
    output_ms_case_1 = net(x, 5, 20)
    expect_output_case_1 = np.clip(x_np, 5, 20)
    output_ms_case_2 = net(x, 20, 5)
    expect_output_case_2 = np.clip(x_np, 20, 5)
    assert np.allclose(output_ms_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_ms_case_2.asnumpy(), expect_output_case_2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clamp_dtype_promotion(mode):
    """
    Feature: test Tensor.clamp dtype promotion for both ClampScalar and ClampTensor
    Description: When x is int32 and the bounds are floating (float scalar or
                 float32 tensor), the output dtype must be promoted to a floating
                 type, matching ClampScalarFuncImpl::InferType (via
                 ops::ClampScalarPromoteType) and ClampTensorFuncImpl::InferType
                 (via ops::ClampTensorPromoteType). Guards both the pyboost
                 customize calls and the fallback IR builders
                 REG_FALLBACK_BUILDER("ClampScalar") / REG_FALLBACK_BUILDER("ClampTensor")
                 against the regression where bounds were cast down to the input
                 dtype (which would have rounded 3.5/20.5 to 3/20 and lost precision).
    Expectation: output dtype is floating and values equal np.clip on promoted types,
                 for all three combinations (both bounds / only min / only max) on
                 both the scalar and the tensor dispatch paths.
    """
    ms.set_context(mode=mode)
    x_np = np.array([[1, 25, 5, 7], [4, 11, 6, 21]], dtype=np.int32)
    x = Tensor(x_np)
    x_f32 = x_np.astype(np.float32)
    net = ClampNet()

    # ---- ClampScalar path: int32 x with float scalar bounds ----
    # Without proper promotion 3.5 / 20.5 would be cast to 3 / 20 and clip results
    # would be off by 0.5 at the boundary values.
    def assert_float(out, label):
        assert out.asnumpy().dtype.kind == 'f', \
            f"{label}: expected floating output, got {out.dtype}"

    output_s_both = net(x, 3.5, 20.5)
    assert_float(output_s_both, "scalar both")
    assert np.allclose(output_s_both.asnumpy(), np.clip(x_f32, 3.5, 20.5))

    output_s_min = net(x, 3.5, None)
    assert_float(output_s_min, "scalar min-only")
    assert np.allclose(output_s_min.asnumpy(), np.maximum(x_f32, 3.5))

    output_s_max = net(x, None, 20.5)
    assert_float(output_s_max, "scalar max-only")
    assert np.allclose(output_s_max.asnumpy(), np.minimum(x_f32, 20.5))

    # ---- ClampTensor path: int32 x with float32 tensor bounds ----
    min_np = np.array([3.5, 3.5, 3.5, 3.5], dtype=np.float32)
    max_np = np.array([20.5, 20.5, 20.5, 20.5], dtype=np.float32)
    min_t = Tensor(min_np)
    max_t = Tensor(max_np)

    output_t_both = net(x, min_t, max_t)
    assert output_t_both.dtype == ms.float32, \
        f"tensor both: expected float32, got {output_t_both.dtype}"
    assert np.allclose(output_t_both.asnumpy(), np.clip(x_f32, min_np, max_np))

    output_t_min = net(x, min_t, None)
    assert output_t_min.dtype == ms.float32, \
        f"tensor min-only: expected float32, got {output_t_min.dtype}"
    assert np.allclose(output_t_min.asnumpy(), np.maximum(x_f32, min_np))

    output_t_max = net(x, None, max_t)
    assert output_t_max.dtype == ms.float32, \
        f"tensor max-only: expected float32, got {output_t_max.dtype}"
    assert np.allclose(output_t_max.asnumpy(), np.minimum(x_f32, max_np))
