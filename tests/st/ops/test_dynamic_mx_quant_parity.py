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
"""Dynamic MX Quant cross-framework parity tests.

CANN 9.1 supports these FP8 Dynamic MX Quant APIs, but their device kernels
require Ascend 950 hardware. The 64 runtime cases are retained and explicitly
skipped until they can run on that hardware.
"""

import functools
import os

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
INPUT_SHAPE = (2, 64, 64)
ROUND_MODE = "rint"
BLOCK_SIZE = 32
SKIP_REASON = "Need Ascend950 hardware."

MS_DYNAMIC_MX_QUANT = ops.auto_generate.dynamic_mx_quant
MS_DYNAMIC_MX_QUANT_DUAL = ops.auto_generate.dynamic_mx_quant_with_dual_axis

INPUT_DTYPES = (
    pytest.param(ms.float16, "float16", id="float16"),
    pytest.param(ms.bfloat16, "bfloat16", id="bfloat16"),
)

DESTINATION_DTYPES = (
    pytest.param(23, ms.float8_e5m2, "float8_e5m2", id="fp8_e5m2"),
    pytest.param(24, ms.float8_e4m3fn, "float8_e4m3fn", id="fp8_e4m3fn"),
    pytest.param(291, ms.float8_e5m2, "float8_e5m2", id="fp8_e5m2_alias"),
    pytest.param(292, ms.float8_e4m3fn, "float8_e4m3fn", id="fp8_e4m3fn_alias"),
)

SCALE_CASES = (
    pytest.param(0, 0.0, id="scale_alg_0-dst_type_max_0"),
    pytest.param(1, 0.0, id="scale_alg_1-dst_type_max_0"),
    pytest.param(1, 8.0, id="scale_alg_1-dst_type_max_8"),
)

DUAL_SCALE_ALGS = (
    pytest.param(0, id="scale_alg_0"),
    pytest.param(1, id="scale_alg_1"),
)

AXES = (
    pytest.param(-1, id="last_axis"),
    pytest.param(-2, id="penultimate_axis"),
)


@functools.lru_cache(maxsize=1)
def _configure_runtime(torch_npu):
    """Configure both frameworks when the Ascend 950 suite is enabled."""
    if not torch_npu.npu.is_available():
        pytest.skip("torch-npu reports no available NPU")

    device_name = torch_npu.npu.get_device_name(DEVICE_ID)
    normalized_device_name = "".join(character for character in device_name.lower() if character.isalnum())
    if "ascend950" not in normalized_device_name:
        pytest.skip(f"Dynamic MX Quant parity requires Ascend 950; device {DEVICE_ID} is {device_name}")

    torch_npu.npu.set_device(DEVICE_ID)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend", DEVICE_ID)


def _make_inputs(ms_dtype, torch_dtype, torch):
    """Create identical deterministic inputs in both frameworks."""
    element_count = int(np.prod(INPUT_SHAPE))
    indices = np.arange(element_count, dtype=np.float32).reshape(INPUT_SHAPE)
    values = np.sin(indices * np.float32(0.03125)) * np.float32(6.0)
    ms_input = Tensor(values, ms_dtype)
    torch_input = torch.from_numpy(values).to(torch_dtype).to(f"npu:{DEVICE_ID}")
    return ms_input, torch_input


def _torch_schema_parameters(torch, op_name):
    """Return argument names from one registered torch-npu schema."""
    overload = getattr(getattr(torch.ops.npu, op_name), "default")
    schema = getattr(overload, "_schema")
    return tuple(argument.name for argument in schema.arguments)


def _compare_quantized(ms_tensor, torch_tensor, expected_ms_dtype, expected_torch_dtype):
    """Compare one FP8 output after lossless conversion to float32."""
    assert tuple(ms_tensor.shape) == tuple(torch_tensor.shape) == INPUT_SHAPE
    assert ms_tensor.dtype == expected_ms_dtype
    assert torch_tensor.dtype == expected_torch_dtype
    ms_values = ms_tensor.astype(ms.float32).asnumpy()
    torch_values = torch_tensor.float().cpu().numpy()
    if not np.isfinite(ms_values).all():
        pytest.fail("MindSpore Dynamic MX Quant output contains NaN or infinity")
    if not np.isfinite(torch_values).all():
        pytest.fail("torch-npu Dynamic MX Quant output contains NaN or infinity")
    np.testing.assert_array_equal(ms_values, torch_values)


def _compare_scale(ms_tensor, torch_tensor):
    """Compare one public uint8 E8M0 scale output exactly."""
    assert tuple(ms_tensor.shape) == tuple(torch_tensor.shape)
    assert ms_tensor.dtype == ms.uint8
    assert torch_tensor.dtype == torch.uint8
    np.testing.assert_array_equal(ms_tensor.asnumpy(), torch_tensor.cpu().numpy())


@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("ms_input_dtype,torch_input_dtype_name", INPUT_DTYPES)
@pytest.mark.parametrize("dst_type,ms_output_dtype,torch_output_dtype_name", DESTINATION_DTYPES)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("scale_alg,dst_type_max", SCALE_CASES)
def test_dynamic_mx_quant(
    ms_input_dtype,
    torch_input_dtype_name,
    dst_type,
    ms_output_dtype,
    torch_output_dtype_name,
    axis,
    scale_alg,
    dst_type_max,
):
    """Feature: Dynamic MX Quant single-axis FP8 parity.

    Description: Compare FP8 values and E8M0 scale bytes for all CANN V2 cases.
    Expectation: MindSpore and torch-npu outputs are identical.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    ms_input, torch_input = _make_inputs(ms_input_dtype, getattr(torch, torch_input_dtype_name), torch)
    torch_kwargs = {
        "axis": axis,
        "round_mode": ROUND_MODE,
        "dst_type": dst_type,
        "block_size": BLOCK_SIZE,
        "scale_alg": scale_alg,
        "dst_type_max": dst_type_max,
    }
    if "max_low_bound" in _torch_schema_parameters(torch, "npu_dynamic_mx_quant"):
        torch_kwargs["max_low_bound"] = 0.0
    torch_y, torch_scale = torch_npu.npu_dynamic_mx_quant(torch_input, **torch_kwargs)
    torch_npu.npu.synchronize()
    ms_y, ms_scale = MS_DYNAMIC_MX_QUANT(
        ms_input,
        axis=axis,
        round_mode=ROUND_MODE,
        dst_type=dst_type,
        block_size=BLOCK_SIZE,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
        max_low_bound=0.0,
    )

    _compare_quantized(ms_y, torch_y, ms_output_dtype, getattr(torch, torch_output_dtype_name))
    _compare_scale(ms_scale, torch_scale)


@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("ms_input_dtype,torch_input_dtype_name", INPUT_DTYPES)
@pytest.mark.parametrize("dst_type,ms_output_dtype,torch_output_dtype_name", DESTINATION_DTYPES)
@pytest.mark.parametrize("scale_alg", DUAL_SCALE_ALGS)
def test_dynamic_mx_quant_with_dual_axis(
    ms_input_dtype,
    torch_input_dtype_name,
    dst_type,
    ms_output_dtype,
    torch_output_dtype_name,
    scale_alg,
):
    """Feature: Dynamic MX Quant dual-axis FP8 parity.

    Description: Compare both FP8 outputs and E8M0 scale pairs for CANN V2.
    Expectation: MindSpore and torch-npu outputs are identical.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    ms_input, torch_input = _make_inputs(ms_input_dtype, getattr(torch, torch_input_dtype_name), torch)
    torch_outputs = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        torch_input,
        round_mode=ROUND_MODE,
        dst_type=dst_type,
        scale_alg=scale_alg,
    )
    torch_npu.npu.synchronize()
    ms_outputs = MS_DYNAMIC_MX_QUANT_DUAL(
        ms_input,
        round_mode=ROUND_MODE,
        dst_type=dst_type,
        scale_alg=scale_alg,
    )

    _compare_quantized(ms_outputs[0], torch_outputs[0], ms_output_dtype, getattr(torch, torch_output_dtype_name))
    _compare_scale(ms_outputs[1], torch_outputs[1])
    _compare_quantized(ms_outputs[2], torch_outputs[2], ms_output_dtype, getattr(torch, torch_output_dtype_name))
    _compare_scale(ms_outputs[3], torch_outputs[3])
