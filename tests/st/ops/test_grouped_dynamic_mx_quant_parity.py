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
"""Grouped Dynamic MX Quant cross-framework parity tests.

The runtime cases cover the supported FP8 destinations, input dtypes, scale
algorithms, and group layouts. They remain skipped until Ascend 950 hardware
is available.
"""

import functools
import inspect
import os

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
INPUT_SHAPE = (128, 64)
ROUND_MODE = "rint"
BLOCKSIZE = 32
SKIP_REASON = "Require Ascend 950 Hardware."

INPUT_DTYPES = (
    pytest.param(ms.float16, "float16", id="float16"),
    pytest.param(ms.bfloat16, "bfloat16", id="bfloat16"),
)

DESTINATION_DTYPES = (
    pytest.param(23, ms.float8_e5m2, "float8_e5m2", id="fp8_e5m2"),
    pytest.param(24, ms.float8_e4m3fn, "float8_e4m3fn", id="fp8_e4m3fn"),
)

SCALE_CASES = (
    pytest.param(0, 0.0, id="scale_alg_0-dst_type_max_0"),
    pytest.param(1, 0.0, id="scale_alg_1-dst_type_max_0"),
    pytest.param(1, 8.0, id="scale_alg_1-dst_type_max_8"),
)

GROUP_LAYOUTS = (
    pytest.param((128,), id="one_group"),
    pytest.param((64, 128), id="two_groups"),
)

BASE_PARAMETERS = ("x", "group_index", "round_mode", "dst_type", "blocksize")
V2_PARAMETERS = BASE_PARAMETERS + ("scale_alg", "dst_type_max")


def _ms_grouped_dynamic_mx_quant():
    """Return the generated MindSpore operator."""
    return ops.auto_generate.grouped_dynamic_mx_quant


def _torch_schema_parameters(torch):
    """Return argument names from the registered op-plugin schema."""
    overload = torch.ops.npu.npu_grouped_dynamic_mx_quant.default
    schema = getattr(overload, "_schema")
    return tuple(argument.name for argument in schema.arguments)


@functools.lru_cache(maxsize=1)
def _configure_runtime(torch_npu):
    """Configure both frameworks when the Ascend 950 suite is enabled."""
    if not torch_npu.npu.is_available():
        pytest.skip("torch-npu reports no available NPU")

    device_name = torch_npu.npu.get_device_name(DEVICE_ID)
    normalized_device_name = "".join(character for character in device_name.lower() if character.isalnum())
    if "ascend950" not in normalized_device_name:
        pytest.skip(f"Grouped Dynamic MX Quant parity requires Ascend 950; device {DEVICE_ID} is {device_name}")

    torch_npu.npu.set_device(DEVICE_ID)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend", DEVICE_ID)


def _make_inputs(ms_dtype, torch_dtype, group_layout, torch):
    """Create identical deterministic data and group boundaries."""
    element_count = int(np.prod(INPUT_SHAPE))
    indices = np.arange(element_count, dtype=np.float32).reshape(INPUT_SHAPE)
    values = np.sin(indices * np.float32(0.03125)) * np.float32(6.0)
    group_values = np.asarray(group_layout, dtype=np.int32)
    ms_x = Tensor(values, ms_dtype)
    ms_group_index = Tensor(group_values, ms.int32)
    torch_x = torch.from_numpy(values).to(torch_dtype).to(f"npu:{DEVICE_ID}")
    torch_group_index = torch.from_numpy(group_values).to(torch.int32).to(f"npu:{DEVICE_ID}")
    return ms_x, ms_group_index, torch_x, torch_group_index


def _compare_quantized(ms_tensor, torch_tensor, expected_ms_dtype, expected_torch_dtype):
    """Compare the FP8 output after lossless conversion to float32."""
    assert tuple(ms_tensor.shape) == tuple(torch_tensor.shape) == INPUT_SHAPE
    assert ms_tensor.dtype == expected_ms_dtype
    assert torch_tensor.dtype == expected_torch_dtype
    ms_values = ms_tensor.astype(ms.float32).asnumpy()
    torch_values = torch_tensor.float().cpu().numpy()
    if not np.isfinite(ms_values).all():
        pytest.fail("MindSpore Grouped Dynamic MX Quant output contains NaN or infinity")
    if not np.isfinite(torch_values).all():
        pytest.fail("torch-npu Grouped Dynamic MX Quant output contains NaN or infinity")
    np.testing.assert_array_equal(ms_values, torch_values)


def _compare_scale(ms_tensor, torch_tensor, group_count):
    """Compare the public uint8 E8M0 scale output exactly."""
    expected_shape = (INPUT_SHAPE[0] // 64 + group_count, INPUT_SHAPE[1], 2)
    assert tuple(ms_tensor.shape) == tuple(torch_tensor.shape) == expected_shape
    assert ms_tensor.dtype == ms.uint8
    assert torch_tensor.dtype == torch.uint8
    np.testing.assert_array_equal(ms_tensor.asnumpy(), torch_tensor.cpu().numpy())


@pytest.mark.skip(reason=SKIP_REASON)
def test_grouped_dynamic_mx_quant_frontend_signature():
    """Feature: Grouped Dynamic MX Quant frontend signature.

    Description: Check the MindSpore signature against supported op-plugin schemas.
    Expectation: The full V2 MindSpore signature and compatible torch schema are exposed.
    """
    torch = pytest.importorskip("torch")
    assert tuple(inspect.signature(_ms_grouped_dynamic_mx_quant()).parameters) == V2_PARAMETERS
    assert _torch_schema_parameters(torch) in (BASE_PARAMETERS, V2_PARAMETERS)


@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("ms_input_dtype,torch_input_dtype_name", INPUT_DTYPES)
@pytest.mark.parametrize("dst_type,ms_output_dtype,torch_output_dtype_name", DESTINATION_DTYPES)
@pytest.mark.parametrize("scale_alg,dst_type_max", SCALE_CASES)
@pytest.mark.parametrize("group_layout", GROUP_LAYOUTS)
# Pytest injects the parametrized values into this test function.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def test_grouped_dynamic_mx_quant(
    ms_input_dtype,
    torch_input_dtype_name,
    dst_type,
    ms_output_dtype,
    torch_output_dtype_name,
    scale_alg,
    dst_type_max,
    group_layout,
):
    """Feature: Grouped Dynamic MX Quant FP8 parity.

    Description: Compare FP8 values and E8M0 scale bytes for supported grouped cases.
    Expectation: MindSpore and torch-npu outputs are identical.
    """
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    _configure_runtime(torch_npu)
    ms_x, ms_group_index, torch_x, torch_group_index = _make_inputs(
        ms_input_dtype, getattr(torch, torch_input_dtype_name), group_layout, torch
    )
    torch_kwargs = {
        "round_mode": ROUND_MODE,
        "dst_type": dst_type,
        "blocksize": BLOCKSIZE,
    }
    if _torch_schema_parameters(torch) == V2_PARAMETERS:
        torch_kwargs.update(scale_alg=scale_alg, dst_type_max=dst_type_max)
    elif scale_alg != 0 or dst_type_max != 0.0:
        pytest.skip("installed op-plugin does not expose the V2 scalar arguments")
    torch_op = getattr(torch_npu, "npu_grouped_dynamic_mx_quant")
    torch_y, torch_scale = torch_op(torch_x, torch_group_index, **torch_kwargs)
    torch_npu.npu.synchronize()
    ms_y, ms_scale = _ms_grouped_dynamic_mx_quant()(
        ms_x,
        ms_group_index,
        round_mode=ROUND_MODE,
        dst_type=dst_type,
        blocksize=BLOCKSIZE,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )

    _compare_quantized(ms_y, torch_y, ms_output_dtype, getattr(torch, torch_output_dtype_name))
    _compare_scale(ms_scale, torch_scale, len(group_layout))
