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
"""Grouped Matmul V5 cross-framework parity tests.

The runtime cases cover floating-point, INT8, and native FP8 grouped matrix
inputs. They remain skipped until Ascend 950 hardware is available.
"""

import functools
import inspect
import os

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
GROUP_COUNT = 3
M = 48
K = 64
N = 32
GROUP_SIZES = (8, 16, 24)
GROUP_ENDS = tuple(np.cumsum(GROUP_SIZES).tolist())
FP8_M = 16
FP8_K = 256
SKIP_REASON = "Require Ascend 950 Hardware."

MS_PARAMETERS = (
    "x",
    "weight",
    "bias",
    "scale",
    "offset",
    "antiquant_scale",
    "antiquant_offset",
    "per_token_scale",
    "group_list",
    "activation_input",
    "activation_quant_scale",
    "activation_quant_offset",
    "split_item",
    "group_type",
    "group_list_type",
    "act_type",
    "tuning_config",
    "output_dtype",
    "scale_dtype",
    "per_token_scale_dtype",
)
TORCH_ONLY_PARAMETERS = {"x_dtype", "weight_dtype"}

DENSE_DTYPES = (
    pytest.param(ms.float16, "float16", ms.float16, "float16", 1e-3, 1e-3, id="float16"),
    pytest.param(ms.bfloat16, "bfloat16", ms.float32, "float32", 1e-2, 1e-2, id="bfloat16"),
    pytest.param(ms.float32, "float32", ms.float32, "float32", 1e-4, 1e-4, id="float32"),
)

GROUP_LIST_CASES = (
    pytest.param(GROUP_ENDS, 0, id="cumulative_group_list"),
    pytest.param(GROUP_SIZES, 1, id="group_size_list"),
)

QUANT_CASES = (
    pytest.param(ms.float32, "float32", ms.float16, "float16", 1e-3, 1e-3,
                 id="int8-float32_scale-float16_output"),
    pytest.param(ms.bfloat16, "bfloat16", ms.bfloat16, "bfloat16", 1e-2, 1e-2,
                 id="int8-bfloat16_scale-bfloat16_output"),
)

FP8_DTYPES = (
    pytest.param(ms.float8_e4m3fn, "float8_e4m3fn", id="float8_e4m3fn"),
    pytest.param(ms.float8_e5m2, "float8_e5m2", id="float8_e5m2"),
)


def _ms_grouped_matmul_v5():
    """Return the generated MindSpore operator."""
    return ops.auto_generate.grouped_matmul_v5


def _torch_schema_parameters(torch):
    """Return argument names from the registered op-plugin schema."""
    schema = getattr(torch.ops.npu.npu_grouped_matmul.default, "_schema")
    return tuple(argument.name for argument in schema.arguments)


@functools.lru_cache(maxsize=1)
def _configure_runtime(torch_npu):
    """Configure both frameworks when the Ascend 950 suite is enabled."""
    if not torch_npu.npu.is_available():
        pytest.skip("torch-npu reports no available NPU")

    device_name = torch_npu.npu.get_device_name(DEVICE_ID)
    normalized_device_name = "".join(character for character in device_name.lower() if character.isalnum())
    if "ascend950" not in normalized_device_name:
        pytest.skip(f"Grouped Matmul V5 parity requires Ascend 950; device {DEVICE_ID} is {device_name}")

    torch_npu.npu.set_device(DEVICE_ID)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend", DEVICE_ID)


def _float_values(shape, phase, amplitude=0.25):
    """Create bounded deterministic values in float32 storage."""
    element_count = int(np.prod(shape))
    indices = np.arange(element_count, dtype=np.float32).reshape(shape)
    return np.sin(indices * np.float32(0.03125) + np.float32(phase)) * np.float32(amplitude)


def _tensor_pair(values, ms_dtype, torch_dtype, torch):
    """Create one tensor with matching logical values in both frameworks."""
    ms_tensor = Tensor(values, ms_dtype)
    torch_tensor = torch.from_numpy(values).to(torch_dtype).to(f"npu:{DEVICE_ID}")
    return ms_tensor, torch_tensor


def _tensor_list_pair(values, ms_dtype, torch_dtype, torch):
    """Create matching MindSpore tuples and torch lists."""
    pairs = [_tensor_pair(item, ms_dtype, torch_dtype, torch) for item in values]
    return tuple(pair[0] for pair in pairs), [pair[1] for pair in pairs]


def _group_list_pair(group_values, torch):
    """Create matching int64 group-list tensors."""
    values = np.asarray(group_values, dtype=np.int64)
    return _tensor_pair(values, ms.int64, torch.int64, torch)


def _compare_outputs(ms_outputs, torch_outputs, expected_ms_dtype, expected_torch_dtype, rtol, atol):
    """Compare output count, metadata, finiteness, and numerical values."""
    assert len(ms_outputs) == len(torch_outputs)
    for ms_output, torch_output in zip(ms_outputs, torch_outputs):
        assert tuple(ms_output.shape) == tuple(torch_output.shape)
        assert ms_output.dtype == expected_ms_dtype
        assert torch_output.dtype == expected_torch_dtype
        ms_values = ms_output.astype(ms.float32).asnumpy()
        torch_values = torch_output.float().cpu().numpy()
        if not np.isfinite(ms_values).all():
            pytest.fail("MindSpore GroupedMatmulV5 output contains NaN or infinity")
        if not np.isfinite(torch_values).all():
            pytest.fail("torch-npu Grouped Matmul output contains NaN or infinity")
        np.testing.assert_allclose(ms_values, torch_values, rtol=rtol, atol=atol)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _run_grouped_case(
    ms_x,
    torch_x,
    ms_weight,
    torch_weight,
    ms_bias,
    torch_bias,
    ms_group_list,
    torch_group_list,
    group_list_type,
    expected_ms_dtype,
    expected_torch_dtype,
    rtol,
    atol,
    torch_npu,
    ms_scale=None,
    torch_scale=None,
    ms_output_dtype=None,
    torch_output_dtype=None,
):
    """Run one M-axis grouped case through both frontends."""
    torch_outputs = torch_npu.npu_grouped_matmul(
        torch_x,
        torch_weight,
        bias=torch_bias,
        scale=torch_scale,
        group_list=torch_group_list,
        split_item=3,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=torch_output_dtype,
    )
    torch_npu.npu.synchronize()
    ms_outputs = _ms_grouped_matmul_v5()(
        ms_x,
        ms_weight,
        bias=ms_bias,
        scale=ms_scale,
        group_list=ms_group_list,
        split_item=3,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=ms_output_dtype,
    )
    _compare_outputs(ms_outputs, torch_outputs, expected_ms_dtype, expected_torch_dtype, rtol, atol)


@pytest.mark.skip(reason=SKIP_REASON)
def test_grouped_matmul_v5_frontend_signature():
    """Feature: Grouped Matmul V5 frontend signature.

    Description: Check MindSpore argument ordering against the op-plugin schema.
    Expectation: All shared parameters have identical ordering.
    """
    torch = pytest.importorskip("torch")
    assert tuple(inspect.signature(_ms_grouped_matmul_v5()).parameters) == MS_PARAMETERS
    torch_parameters = _torch_schema_parameters(torch)
    shared_torch_parameters = tuple(
        parameter for parameter in torch_parameters if parameter not in TORCH_ONLY_PARAMETERS
    )
    assert shared_torch_parameters == MS_PARAMETERS


@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize(
    "ms_input_dtype,torch_input_dtype_name,ms_bias_dtype,torch_bias_dtype_name,rtol,atol",
    DENSE_DTYPES,
)
@pytest.mark.parametrize("group_values,group_list_type", GROUP_LIST_CASES)
# Pytest injects the parametrized values into this test function.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def test_dense_grouped_weight(
    ms_input_dtype,
    torch_input_dtype_name,
    ms_bias_dtype,
    torch_bias_dtype_name,
    rtol,
    atol,
    group_values,
    group_list_type,
):
    """Feature: Grouped Matmul V5 floating-point parity.

    Description: Compare floating dtypes with cumulative and per-group-size lists.
    Expectation: MindSpore and torch-npu outputs are numerically equivalent.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    torch_input_dtype = getattr(torch, torch_input_dtype_name)
    torch_bias_dtype = getattr(torch, torch_bias_dtype_name)
    ms_x, torch_x = _tensor_list_pair([_float_values((M, K), phase=0.125)], ms_input_dtype,
                                           torch_input_dtype, torch)
    ms_weight, torch_weight = _tensor_list_pair(
        [_float_values((GROUP_COUNT, K, N), phase=0.5)], ms_input_dtype, torch_input_dtype, torch
    )
    ms_bias, torch_bias = _tensor_list_pair(
        [_float_values((GROUP_COUNT, N), phase=0.75, amplitude=0.1)], ms_bias_dtype, torch_bias_dtype, torch
    )
    ms_group_list, torch_group_list = _group_list_pair(group_values, torch)
    _run_grouped_case(
        ms_x,
        torch_x,
        ms_weight,
        torch_weight,
        ms_bias,
        torch_bias,
        ms_group_list,
        torch_group_list,
        group_list_type,
        ms_input_dtype,
        torch_input_dtype,
        rtol,
        atol,
        torch_npu,
    )


@pytest.mark.skip(reason=SKIP_REASON)
def test_dense_separate_tensor_lists():
    """Feature: Grouped Matmul V5 TensorList parity.

    Description: Compare the split-item-0 multiple-input and multiple-output form.
    Expectation: MindSpore and torch-npu outputs are numerically equivalent.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    x_values = [_float_values((8, K), 0.25), _float_values((16, K), 0.5)]
    weight_values = [_float_values((K, N), 0.75), _float_values((K, N + 16), 1.0)]
    bias_values = [_float_values((N,), 1.25, 0.1), _float_values((N + 16,), 1.5, 0.1)]
    ms_x, torch_x = _tensor_list_pair(x_values, ms.float16, torch.float16, torch)
    ms_weight, torch_weight = _tensor_list_pair(weight_values, ms.float16, torch.float16, torch)
    ms_bias, torch_bias = _tensor_list_pair(bias_values, ms.float16, torch.float16, torch)

    torch_outputs = torch_npu.npu_grouped_matmul(
        torch_x,
        torch_weight,
        bias=torch_bias,
        split_item=0,
        group_type=-1,
    )
    torch_npu.npu.synchronize()
    ms_outputs = _ms_grouped_matmul_v5()(
        ms_x,
        ms_weight,
        bias=ms_bias,
        split_item=0,
        group_type=-1,
    )
    _compare_outputs(ms_outputs, torch_outputs, ms.float16, torch.float16, 1e-3, 1e-3)


@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize(
    "ms_scale_dtype,torch_scale_dtype_name,ms_output_dtype,torch_output_dtype_name,rtol,atol",
    QUANT_CASES,
)
# Pytest injects the parametrized values into this test function.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def test_int8_grouped_weight(
    ms_scale_dtype,
    torch_scale_dtype_name,
    ms_output_dtype,
    torch_output_dtype_name,
    rtol,
    atol,
):
    """Feature: Grouped Matmul V5 INT8 parity.

    Description: Compare INT8 inputs with floating dequantized outputs.
    Expectation: MindSpore and torch-npu outputs are numerically equivalent.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    torch_scale_dtype = getattr(torch, torch_scale_dtype_name)
    torch_output_dtype = getattr(torch, torch_output_dtype_name)
    rng = np.random.default_rng(29)
    x_values = [rng.integers(-3, 4, size=(M, K), dtype=np.int8)]
    weight_values = [rng.integers(-3, 4, size=(K, N), dtype=np.int8) for _ in range(GROUP_COUNT)]
    bias_values = [rng.integers(-5, 6, size=(N,), dtype=np.int32) for _ in range(GROUP_COUNT)]
    scale_values = [np.linspace(0.005, 0.015, N, dtype=np.float32) for _ in range(GROUP_COUNT)]
    ms_x, torch_x = _tensor_list_pair(x_values, ms.int8, torch.int8, torch)
    ms_weight, torch_weight = _tensor_list_pair(weight_values, ms.int8, torch.int8, torch)
    ms_bias, torch_bias = _tensor_list_pair(bias_values, ms.int32, torch.int32, torch)
    ms_scale, torch_scale = _tensor_list_pair(scale_values, ms_scale_dtype, torch_scale_dtype, torch)
    ms_group_list, torch_group_list = _group_list_pair(GROUP_ENDS, torch)
    _run_grouped_case(
        ms_x,
        torch_x,
        ms_weight,
        torch_weight,
        ms_bias,
        torch_bias,
        ms_group_list,
        torch_group_list,
        0,
        ms_output_dtype,
        torch_output_dtype,
        rtol,
        atol,
        torch_npu,
        ms_scale=ms_scale,
        torch_scale=torch_scale,
        ms_output_dtype=ms_output_dtype,
        torch_output_dtype=torch_output_dtype,
    )


@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("ms_input_dtype,torch_input_dtype_name", FP8_DTYPES)
def test_fp8_grouped_weight(ms_input_dtype, torch_input_dtype_name):
    """Feature: Grouped Matmul V5 native FP8 parity.

    Description: Compare both supported native FP8 matrix-input types.
    Expectation: MindSpore and torch-npu outputs are numerically equivalent.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    torch_input_dtype = getattr(torch, torch_input_dtype_name)
    representable = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    x_values = np.resize(representable, (FP8_M, FP8_K))
    weight_values = np.resize(representable[::-1], (1, FP8_K, N))
    scale_values = np.ones((1, N), dtype=np.float32)

    ms_x, torch_x = _tensor_list_pair([x_values], ms_input_dtype, torch_input_dtype, torch)
    ms_weight, torch_weight = _tensor_list_pair([weight_values], ms_input_dtype, torch_input_dtype, torch)
    ms_scale, torch_scale = _tensor_list_pair([scale_values], ms.float32, torch.float32, torch)
    ms_group_list, torch_group_list = _group_list_pair((FP8_M,), torch)

    torch_outputs = torch_npu.npu_grouped_matmul(
        torch_x,
        torch_weight,
        scale=torch_scale,
        group_list=torch_group_list,
        split_item=2,
        group_type=0,
        group_list_type=0,
        output_dtype=torch.float32,
    )
    torch_npu.npu.synchronize()
    ms_outputs = _ms_grouped_matmul_v5()(
        ms_x,
        ms_weight,
        scale=ms_scale,
        group_list=ms_group_list,
        split_item=2,
        group_type=0,
        group_list_type=0,
        output_dtype=ms.float32,
    )
    _compare_outputs(ms_outputs, torch_outputs, ms.float32, torch.float32, 1e-3, 1e-3)
