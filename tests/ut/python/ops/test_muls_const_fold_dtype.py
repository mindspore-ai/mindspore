# Copyright 2026 Huawei Technologies Co., Ltd
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
"""UT for Muls constant folding.

Focuses on dtype consistency and float64 precision in constant folding.
"""

import numpy as np

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops.auto_generate import muls_op
import pytest


class ConstMulsNet(nn.Cell):
    def __init__(self):
        super().__init__()
        # Make both inputs constants in graph to trigger constant folding.
        self.x = Tensor(np.array([1.0, -2.0, 3.5], np.float32), ms.float32)
        # Use np.float64 to ensure scalar dtype is float64.
        self.y = np.float64(2.0)

    def construct(self):
        return muls_op(self.x, self.y)


def test_muls_const_fold_keep_tensor_dtype():
    """
    Feature: Muls constant folding.
    Description: In GRAPH_MODE, when both Tensor and Scalar are constants,
                 constant folding should keep dtype consistent with InferType
                 (PromoteTensorScalarType), i.e. float32 tensor
                 * float64 scalar -> float32.
    Expectation: Output dtype is float32.
    """
    context.set_context(mode=context.GRAPH_MODE)
    out = ConstMulsNet()()
    assert out.dtype == ms.float32


class ConstMulsBFloat16Net(nn.Cell):
    def __init__(self):
        super().__init__()
        # Make both inputs constants in graph to trigger constant folding.
        self.x = Tensor(np.array([1.25, -2.5, 3.75], np.float32), ms.bfloat16)
        # Use float64 scalar with non-trivial rounding.
        self.y = np.float64(0.1)

    def construct(self):
        return muls_op(self.x, self.y)


def _bf16_round_to_f32(x):
    """Round float32 values to bfloat16, then convert back to float32."""
    x_f32 = np.asarray(x, np.float32)
    is_scalar = x_f32.ndim == 0
    if is_scalar:
        x_f32 = x_f32.reshape(1)
    u32 = x_f32.view(np.uint32)
    rounding_bias = ((u32 >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    bf16_u16 = ((u32 + rounding_bias) >> 16).astype(np.uint32)
    out = (bf16_u16 << 16).view(np.float32)
    if is_scalar:
        return out[0]
    return out


def test_muls_const_fold_bfloat16_no_extra_precision_loss():
    """
    Feature: Muls constant folding.
    Description: For bfloat16 tensor input, constant folding should be
                 consistent with runtime semantics (no extra precision loss
                 beyond bf16).
    Expectation: GRAPH_MODE (const-fold) output matches bf16 rounding
                 semantics.
    """
    context.set_context(mode=context.GRAPH_MODE)
    out = ConstMulsBFloat16Net()()
    assert out.dtype == ms.bfloat16
    out_np = out.asnumpy().astype(np.float32)

    x_in = np.array([1.25, -2.5, 3.75], np.float32)
    y_in = np.float32(0.1)
    x_bf16 = _bf16_round_to_f32(x_in)
    y_bf16 = _bf16_round_to_f32(y_in)
    expected = _bf16_round_to_f32(x_bf16 * y_bf16)
    np.testing.assert_array_equal(out_np, expected)


class ConstMulsGenericNet(nn.Cell):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def construct(self):
        return muls_op(self.x, self.y)


def _np_expected_muls(x_np, y, out_ms_dtype):
    """Compute expected result in numpy for MindSpore's promotion result dtype.
    """
    # NOTE: For bfloat16, validate in a dedicated test to match bf16 rounding.
    if out_ms_dtype == ms.bfloat16:
        raise ValueError("bfloat16 expected is checked by dedicated test.")
    if out_ms_dtype in (ms.float16, ms.float32, ms.float64):
        np_out_dtype = {
            ms.float16: np.float16,
            ms.float32: np.float32,
            ms.float64: np.float64,
        }[out_ms_dtype]
        return x_np.astype(np_out_dtype) * np_out_dtype(y)
    if out_ms_dtype in (
        ms.int8, ms.int16, ms.int32, ms.int64,
        ms.uint8, ms.uint16, ms.uint32, ms.uint64,
    ):
        np_out_dtype = {
            ms.int8: np.int8,
            ms.int16: np.int16,
            ms.int32: np.int32,
            ms.int64: np.int64,
            ms.uint8: np.uint8,
            ms.uint16: np.uint16,
            ms.uint32: np.uint32,
            ms.uint64: np.uint64,
        }[out_ms_dtype]
        return x_np.astype(np_out_dtype) * np_out_dtype(y)
    if out_ms_dtype == ms.bool_:
        # Muls bool uses logical-and semantics in constant folding.
        return np.logical_and(x_np.astype(np.bool_), bool(y))
    raise ValueError(f"Unsupported out dtype: {out_ms_dtype}")


@pytest.mark.parametrize(
    "x_ms_dtype,x_np_dtype,y,expected_ms_dtype",
    [
        # bool category
        (ms.bool_, np.bool_, True, ms.bool_),
        (ms.bool_, np.bool_, 2, ms.int64),          # bool * int -> int
        (ms.bool_, np.bool_, 0.5, ms.float32),      # bool * float -> float
        # integral category (keep tensor dtype when both integral)
        (ms.int8, np.int8, 2, ms.int8),
        (ms.int32, np.int32, 3, ms.int32),
        (ms.uint8, np.uint8, 4, ms.uint8),
        # int -> float promotion
        (ms.int32, np.int32, 0.5, ms.float32),
        # float category (keep tensor dtype when both float)
        (ms.float16, np.float16, 2.0, ms.float16),
        (ms.float32, np.float32, 2, ms.float32),
        (ms.float64, np.float64, 2, ms.float64),
        # float tensor with numpy float64 scalar keeps tensor dtype
        (ms.float32, np.float32, np.float64(2.0), ms.float32),
    ],
)
def test_muls_const_fold_various_dtypes(
    x_ms_dtype, x_np_dtype, y, expected_ms_dtype
):
    """
    Feature: Muls constant folding.
    Description: In GRAPH_MODE, with constant Tensor and constant Scalar
                 inputs,
                 Muls should be constant-folded and keep dtype consistent with
                 PromoteTensorScalarType.
    Expectation: Output dtype matches expected, and values match for supported
                 dtypes.
    """
    context.set_context(mode=context.GRAPH_MODE)
    if x_ms_dtype == ms.uint8:
        x_np = np.array([1, 0, 2], dtype=x_np_dtype)
    else:
        x_np = np.array([1, 0, -2], dtype=x_np_dtype)
    x = Tensor(x_np, x_ms_dtype)
    out = ConstMulsGenericNet(x, y)()
    assert out.dtype == expected_ms_dtype
    if expected_ms_dtype != ms.bfloat16:
        expected = _np_expected_muls(x_np, y, expected_ms_dtype)
        out_np = out.asnumpy()
        if expected_ms_dtype in (ms.float16, ms.float32, ms.float64):
            np.testing.assert_allclose(out_np, expected, rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(out_np, expected)
