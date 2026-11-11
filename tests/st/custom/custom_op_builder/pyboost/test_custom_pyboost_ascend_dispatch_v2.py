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
"""
Custom Op V2 dispatch verification — compare v2 custom ops with framework/numpy baselines.
Only PYNATIVE_MODE (pyboost). Covers: dtypes, shapes, broadcasting, alpha parameter, multi-output.
"""
# pylint: disable=too-many-function-args

import numpy as np
import mindspore as ms
from mindspore._c_expression import typing
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_close(actual, expected, rtol=1e-4, atol=1e-4):
    """Compare two ms.Tensor values, casting to float32 for safe comparison."""
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} vs {expected.dtype}"
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    a = actual.asnumpy().astype(np.float32)
    e = expected.asnumpy().astype(np.float32)
    assert np.allclose(a, e, rtol=rtol, atol=atol), \
        f"max diff = {np.max(np.abs(a - e))}, rtol={rtol}, atol={atol}"


def _load_op(name, cpp_file):
    return CustomOpBuilder(name, [f'v2_test_files/{cpp_file}'], backend="Ascend").load()


def _load_jit_op(name, cpp_file):
    return CustomOpBuilder(name, [f'jit_test_files/{cpp_file}'], backend="Ascend").load()


def _rand_np(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


def _np_sin(x):
    return np.sin(x).astype(x.dtype)


def _np_abs(x):
    return np.abs(x).astype(x.dtype)


def _np_relu(x):
    return np.maximum(x, 0).astype(x.dtype)


def _np_add(x, y, alpha=1.0):
    return (x + np.asarray(alpha, dtype=x.dtype) * y).astype(x.dtype)


def _np_sub(x, y, alpha=1.0):
    return (x - np.asarray(alpha, dtype=x.dtype) * y).astype(x.dtype)


def _np_mul(x, y):
    return (x * y).astype(x.dtype)


# ---------------------------------------------------------------------------
# Sub-cases
# ---------------------------------------------------------------------------

def _test_unary_dtypes(ops, func_name, np_fn):
    fn = getattr(ops, func_name)
    for dtype, rtol in [(np.float32, 1e-4), (np.float16, 1e-3)]:
        x_np = _rand_np((32, 64), dtype)
        x = ms.Tensor(x_np)
        expected = ms.Tensor(np_fn(x_np))
        _assert_close(fn(x), expected, rtol=rtol, atol=rtol)


def _test_unary_shapes(ops, func_name, np_fn):
    fn = getattr(ops, func_name)
    for shape in [(128,), (32, 64), (2, 3, 4, 5)]:
        x_np = _rand_np(shape)
        x = ms.Tensor(x_np)
        expected = ms.Tensor(np_fn(x_np))
        _assert_close(fn(x), expected)


def _test_binary_basic(ops, func_name, np_fn, has_alpha=False):
    fn = getattr(ops, func_name)
    x_np, y_np = _rand_np((32, 64)), _rand_np((32, 64))
    x, y = ms.Tensor(x_np), ms.Tensor(y_np)
    if has_alpha:
        expected = ms.Tensor(np_fn(x_np, y_np, alpha=1))
        _assert_close(fn(x, y, 1.0), expected)
    else:
        expected = ms.Tensor(np_fn(x_np, y_np))
        _assert_close(fn(x, y), expected)


def _test_binary_alpha(ops, func_name, np_fn):
    fn = getattr(ops, func_name)
    x_np, y_np = _rand_np((32, 64)), _rand_np((32, 64))
    x, y = ms.Tensor(x_np), ms.Tensor(y_np)
    for alpha in [0.5, 2.0]:
        expected = ms.Tensor(np_fn(x_np, y_np, alpha=alpha))
        _assert_close(fn(x, y, alpha), expected)


def _test_binary_broadcast(ops, func_name, np_fn, has_alpha=False):
    """Verify binary custom op broadcasting matches the numerical baseline."""
    fn = getattr(ops, func_name)
    # dim broadcast: (3,1,5) + (1,4,5) -> (3,4,5)
    x_np, y_np = _rand_np((3, 1, 5)), _rand_np((1, 4, 5))
    x, y = ms.Tensor(x_np), ms.Tensor(y_np)
    if has_alpha:
        expected = ms.Tensor(np_fn(x_np, y_np, alpha=1))
        _assert_close(fn(x, y, 1.0), expected)
    else:
        expected = ms.Tensor(np_fn(x_np, y_np))
        _assert_close(fn(x, y), expected)
    # scalar broadcast: (32,64) + (1,) -> (32,64)
    x_np, y_np = _rand_np((32, 64)), _rand_np((1,))
    x, y = ms.Tensor(x_np), ms.Tensor(y_np)
    if has_alpha:
        expected = ms.Tensor(np_fn(x_np, y_np, alpha=1))
        _assert_close(fn(x, y, 1.0), expected)
    else:
        expected = ms.Tensor(np_fn(x_np, y_np))
        _assert_close(fn(x, y), expected)


def _test_binary_dtypes(ops, func_name, np_fn, has_alpha=False):
    """Verify float16 binary custom op outputs."""
    fn = getattr(ops, func_name)
    x_np = _rand_np((32, 64), np.float16)
    y_np = _rand_np((32, 64), np.float16)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    if has_alpha:
        expected = ms.Tensor(np_fn(x_np, y_np, alpha=1))
        _assert_close(fn(x, y, 1.0), expected, rtol=1e-3, atol=1e-3)
    else:
        expected = ms.Tensor(np_fn(x_np, y_np))
        _assert_close(fn(x, y), expected, rtol=1e-3, atol=1e-3)


def _test_batch_norm_outputs(ops):
    """Verify custom batch_norm output value and auxiliary tensors."""
    x_np = (3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32)
    scale_np = np.ones(2).astype(np.float32)
    bias_np = np.ones(2).astype(np.float32)
    mean_np = np.ones(2).astype(np.float32)
    var_np = np.ones(2).astype(np.float32)

    x = ms.Tensor(x_np)
    scale = ms.Tensor(scale_np)
    bias = ms.Tensor(bias_np)
    mean = ms.Tensor(mean_np)
    variance = ms.Tensor(var_np)

    outputs = ops.npu_batch_norm_v2(x, scale, bias, mean, variance, False, 0.1, 1e-5)
    expect = ((x_np - mean_np.reshape(1, 2, 1, 1)) / np.sqrt(var_np.reshape(1, 2, 1, 1) + 1e-5))
    expect = expect * scale_np.reshape(1, 2, 1, 1) + bias_np.reshape(1, 2, 1, 1)

    assert len(outputs) == 3
    _assert_close(outputs[0], ms.Tensor(expect))
    assert outputs[1].shape == (2,)
    assert outputs[2].shape == (2,)
    assert outputs[1].dtype == ms.float32
    assert outputs[2].dtype == ms.float32


def _test_legacy_jit_v2_outputs():
    """Verify legacy jit_test_files custom_v2 samples remain compatible."""
    cast_ops = _load_jit_op("jit_v2_cast_compat", "pyboost_aclnn_cast_v2.cpp")
    x_np = np.random.randn(32, 64).astype(np.float16)
    dst_type_id = typing.type_to_type_id(ms.dtype.float32)
    output_np = cast_ops.npu_cast_v2(ms.Tensor(x_np), dst_type_id).asnumpy()
    assert output_np.dtype == 'float32'
    assert output_np.shape == (32, 64)
    assert np.allclose(output_np, x_np.astype(np.float32), rtol=1e-3, atol=1e-3)

    sin_ops = _load_jit_op("jit_v2_sin_compat", "pyboost_aclnn_sin_v2.cpp")
    x_np = np.random.randn(32, 64).astype(np.float32)
    x = ms.Tensor(x_np)
    _assert_close(sin_ops.npu_sin_v2(x), ms.Tensor(np.sin(x_np)))


def _test_cast_outputs(ops):
    """Verify custom cast output matches Tensor.astype."""
    x_np = _rand_np((32, 64), np.float16)
    x = ms.Tensor(x_np)
    dst_type_id = typing.type_to_type_id(ms.dtype.float32)
    expected = ms.Tensor(x_np.astype(np.float32))
    _assert_close(ops.npu_cast_v2(x, dst_type_id), expected, rtol=1e-3, atol=1e-3)


def _test_reduce_argmin_outputs(sum_ops, argmin_ops):
    """Verify reduce_sum and argmin custom ops match numerical baselines."""
    x_np = _rand_np((4, 5, 6), np.float32)
    x = ms.Tensor(x_np)
    expected_sum = ms.Tensor(np.sum(np.abs(x_np), axis=1, keepdims=True).astype(np.float32))
    custom_sum = sum_ops.npu_abs_reduce_sum_v2(x, (1,), True, None)
    _assert_close(custom_sum, expected_sum)
    try:
        invalid_sum = sum_ops.npu_abs_reduce_sum_v2(x, (x_np.ndim,), True, None)
        invalid_sum.asnumpy()
    except RuntimeError as err:
        assert "out of range" in str(err)
    else:
        raise AssertionError("Expected out-of-range reduce axis to fail")

    x_np = _rand_np((2, 3, 4, 5), np.float32)
    x = ms.Tensor(x_np)
    expected_argmin = ms.Tensor(np.argmin(x_np, axis=0).astype(np.int64))
    custom_argmin = argmin_ops.npu_arg_min_v2(x, 0, False)
    _assert_close(custom_argmin, expected_argmin)


def _test_avg_pool2d_outputs(ops):
    """Verify avg_pool2d custom op matches a numerical baseline."""
    image_np = np.array([[[4.1702e-1, 7.2032e-1, 1.1437e-4, 3.0223e-1],
                          [1.4676e-1, 9.2339e-2, 1.8626e-1, 3.4556e-1],
                          [3.9677e-1, 5.3882e-1, 4.1919e-1, 6.8522e-1],
                          [2.0445e-1, 8.7812e-1, 2.7338e-2, 6.7047e-1]]]).astype(np.float32)
    image = ms.Tensor(image_np)
    padded = np.pad(image_np, ((0, 0), (1, 1), (1, 1)), mode='constant')
    expected = np.empty((1, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            expected[:, i, j] = np.mean(padded[:, i * 2:i * 2 + 2, j * 2:j * 2 + 2], axis=(1, 2))
    expected_out = ms.Tensor(expected)
    custom_out = ops.npu_avgpool2d_v2(image, (2, 2), (2, 2), (1, 1), False, True, 0, False)
    _assert_close(custom_out, expected_out, rtol=1e-4, atol=1e-4)


def _test_quant_batch_matmul_outputs(ops):
    """Verify quant_batch_matmul custom op with FRACTAL_NZ input."""
    batch, m, k, n = 2, 128, 256, 128
    x1 = np.random.randint(-5, 5, size=(batch, m, k)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(batch, k, n)).astype(np.int8)
    scale = np.ones([n]).astype(np.float32)
    expected = np.matmul(x1.astype(np.int32), x2.astype(np.int32)) * scale

    ms_x1 = ms.Tensor(x1)
    ms_x2 = ms.ops.auto_generate.format_cast(ms.Tensor(x2), 29)
    output_dtype = typing.type_to_type_id(ms.dtype.bfloat16)
    output = ops.quant_batch_matmul_v2(ms_x1, ms_x2, ms.Tensor(scale), None, None, None, False, False, "FRACTAL_NZ",
                                       output_dtype)
    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), rtol=1e-2, atol=1e-2)


def _rand_scaled_np(shape, dtype=np.float16, scale=0.1):
    return (np.random.randn(*shape) * scale).astype(dtype)


def _softmax_np(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _layer_norm_np(x, weight, bias, axis):
    mean = np.mean(x.astype(np.float32), axis=axis, keepdims=True)
    variance = np.mean(np.square(x.astype(np.float32) - mean), axis=axis, keepdims=True)
    expected = (x.astype(np.float32) - mean) / np.sqrt(variance + 1e-5)
    return (expected * weight.astype(np.float32) + bias.astype(np.float32)).astype(np.float16)


def _test_llm_mint_ops(ops):
    """Verify Ascend custom LLM ops against numerical baselines."""
    x_np = _rand_scaled_np((16, 32))
    y_np = _rand_scaled_np((32, 24))
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    matmul_expected = ms.Tensor(np.matmul(x_np, y_np).astype(np.float16))
    _assert_close(ops.npu_llm_matmul_v2(x, y), matmul_expected, rtol=1e-3, atol=1e-3)

    normalized_shape = [32]
    ln_x_np = _rand_scaled_np((2, 8, 32))
    weight_np = _rand_scaled_np((32,))
    bias_np = _rand_scaled_np((32,))
    ln_x = ms.Tensor(ln_x_np)
    weight = ms.Tensor(weight_np)
    bias = ms.Tensor(bias_np)
    expected_ln = ms.Tensor(_layer_norm_np(ln_x_np, weight_np, bias_np, axis=-1))
    custom_ln = ops.npu_llm_layer_norm_v2(ln_x, normalized_shape, weight, bias, 1e-5)
    _assert_close(custom_ln, expected_ln, rtol=1e-3, atol=1e-3)

    normalized_shape = [8, 32]
    weight_np = _rand_scaled_np((8, 32))
    bias_np = _rand_scaled_np((8, 32))
    weight = ms.Tensor(weight_np)
    bias = ms.Tensor(bias_np)
    expected_ln = ms.Tensor(_layer_norm_np(ln_x_np, weight_np, bias_np, axis=(-2, -1)))
    custom_ln = ops.npu_llm_layer_norm_v2(ln_x, normalized_shape, weight, bias, 1e-5)
    _assert_close(custom_ln, expected_ln, rtol=1e-3, atol=1e-3)

    batch, heads, seq_len, head_dim = 1, 2, 16, 32
    query_np = _rand_scaled_np((batch, heads, seq_len, head_dim))
    key_np = _rand_scaled_np((batch, heads, seq_len, head_dim))
    value_np = _rand_scaled_np((batch, heads, seq_len, head_dim))
    query = ms.Tensor(query_np)
    key = ms.Tensor(key_np)
    value = ms.Tensor(value_np)
    scale = float(1.0 / np.sqrt(head_dim))
    scores = np.matmul(query_np.astype(np.float32), np.swapaxes(key_np.astype(np.float32), -1, -2)) * scale
    expected_attention_np = np.matmul(_softmax_np(scores, axis=-1), value_np.astype(np.float32)).astype(np.float16)
    expected_attention = ms.Tensor(expected_attention_np)
    custom_attention = ops.npu_llm_flash_attention_v2(query, key, value, scale)
    _assert_close(custom_attention, expected_attention, rtol=5e-2, atol=5e-2)


# ===========================================================================
#  V2 dispatch verification groups
# ===========================================================================

def _test_dispatch_ops_against_mint():
    """Verify dispatch-enabled v2 ACLNN ops against numerical baselines."""
    # --- Unary: sin ---
    sin_ops = _load_op("v2_all_sin", "pyboost_aclnn_sin_v2.cpp")
    _test_unary_dtypes(sin_ops, "npu_sin_v2", _np_sin)
    _test_unary_shapes(sin_ops, "npu_sin_v2", _np_sin)

    # --- Unary: abs ---
    abs_ops = _load_op("v2_all_abs", "pyboost_aclnn_abs_v2.cpp")
    _test_unary_dtypes(abs_ops, "npu_abs_v2", _np_abs)
    _test_unary_shapes(abs_ops, "npu_abs_v2", _np_abs)

    # --- Unary: relu ---
    relu_ops = _load_op("v2_all_relu", "pyboost_aclnn_relu_v2.cpp")
    _test_unary_dtypes(relu_ops, "npu_relu_v2", _np_relu)
    _test_unary_shapes(relu_ops, "npu_relu_v2", _np_relu)

    # --- Binary: add (with alpha) ---
    add_ops = _load_op("v2_all_add", "pyboost_aclnn_add_v2.cpp")
    _test_binary_basic(add_ops, "npu_add_v2", _np_add, has_alpha=True)
    _test_binary_alpha(add_ops, "npu_add_v2", _np_add)
    _test_binary_broadcast(add_ops, "npu_add_v2", _np_add, has_alpha=True)
    _test_binary_dtypes(add_ops, "npu_add_v2", _np_add, has_alpha=True)

    # --- Binary: sub (with alpha) ---
    sub_ops = _load_op("v2_all_sub", "pyboost_aclnn_sub_v2.cpp")
    _test_binary_basic(sub_ops, "npu_sub_v2", _np_sub, has_alpha=True)
    _test_binary_alpha(sub_ops, "npu_sub_v2", _np_sub)
    _test_binary_broadcast(sub_ops, "npu_sub_v2", _np_sub, has_alpha=True)

    # --- Binary: mul (no alpha) ---
    mul_ops = _load_op("v2_all_mul", "pyboost_aclnn_mul_v2.cpp")
    _test_binary_basic(mul_ops, "npu_mul_v2", _np_mul)
    _test_binary_broadcast(mul_ops, "npu_mul_v2", _np_mul)
    _test_binary_dtypes(mul_ops, "npu_mul_v2", _np_mul)

    # --- Multi-output: batch_norm ---
    batch_norm_ops = _load_op("v2_all_batch_norm", "pyboost_aclnn_batch_norm_v2.cpp")
    _test_batch_norm_outputs(batch_norm_ops)


def _test_llm_ops_against_mint():
    """Verify Ascend custom LLM ops against numerical baselines."""
    llm_ops = _load_op("v2_all_llm_ops", "pyboost_aclnn_llm_ops_v2.cpp")
    _test_llm_mint_ops(llm_ops)


def _test_existing_aclnn_ops_against_mint_v2():
    """Verify migrated PyBoost ACLNN examples against numerical baselines."""
    cast_ops = _load_op("v2_all_cast", "pyboost_aclnn_cast_v2.cpp")
    _test_cast_outputs(cast_ops)

    sum_ops = _load_op("v2_all_sum", "pyboost_aclnn_sum_v2.cpp")
    argmin_ops = _load_op("v2_all_argmin", "pyboost_aclnn_argmin_v2.cpp")
    _test_reduce_argmin_outputs(sum_ops, argmin_ops)

    avg_pool_ops = _load_op("v2_all_avg_pool2d", "pyboost_aclnn_avg_pool_2d_v2.cpp")
    _test_avg_pool2d_outputs(avg_pool_ops)

    quant_matmul_ops = _load_op("v2_all_quant_batch_matmul", "pyboost_aclnn_quant_batch_matmul_v2.cpp")
    _test_quant_batch_matmul_outputs(quant_matmul_ops)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=20.0)
def test_ascend_custom_v2_dispatch_and_llm():
    """
    Feature: CustomOpBuilder with stateless custom_v2 API (MS_DISPATCH_ACLNN).
    Description: Verify Ascend custom_v2 dispatch and LLM operators.
    Expectation: All outputs match mint or numerical baselines.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    np.random.seed(0)
    _test_dispatch_ops_against_mint()
    _test_llm_ops_against_mint()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=20.0)
def test_ascend_custom_v2_existing_and_legacy():
    """
    Feature: CustomOpBuilder with stateless custom_v2 API (MS_DISPATCH_ACLNN).
    Description: Verify migrated ACLNN operators and legacy jit_test_files compatibility.
    Expectation: All outputs match numerical baselines, and legacy samples still work.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    np.random.seed(1)
    _test_existing_aclnn_ops_against_mint_v2()
    _test_legacy_jit_v2_outputs()
