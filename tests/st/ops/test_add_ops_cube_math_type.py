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
"""Test cube_math_type configuration for Addmm and Baddbmm."""

import sys

import numpy as np

import mindspore as ms
from mindspore import jit, mint

if __name__ != "__main__":
    import subprocess
    import pytest
    from tests.mark_utils import arg_mark


def addmm(input_tensor, mat1, mat2, beta, alpha):
    """Run Addmm."""
    return mint.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)


def baddbmm(input_tensor, batch1, batch2, beta, alpha):
    """Run Baddbmm."""
    return mint.baddbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)


def run_op(func, inputs, mode):
    """Run one function in PyBoost or KBK mode."""
    if mode == "pynative":
        ms.set_context(mode=ms.PYNATIVE_MODE)
        return func(*inputs)
    return jit(func, backend="ms_backend", jit_level="O0")(*inputs)


def run_addmm_mode0_scalar(mode):
    """Check that KBK preserves non-exact Addmm scalars in mode 0."""
    inputs = tuple(ms.Tensor([[value]], dtype=ms.bfloat16) for value in (4.0, -8.0, -5.0))
    for beta, expected in ((-4.775000095367432, 62.25), (0.0, 81.5)):
        output = run_op(addmm, (*inputs, beta, 2.0315001010894775), mode)
        assert output.shape == (1, 1)
        assert output.dtype == ms.bfloat16
        assert output.float().asnumpy().item() == expected


def run_mode4_broadcast(mode):
    """Check mode 4 when input is broadcast relative to the matmul result."""
    addmm_inputs = (
        np.array([[4.0]], np.float32),
        np.array([[-8.0], [-4.0]], np.float32),
        np.array([[-5.0, 3.0]], np.float32),
    )
    baddbmm_inputs = (
        np.array([4.0, -2.0], np.float32),
        np.array([[[-8.0], [-4.0]], [[2.0], [6.0]]], np.float32),
        np.array([[[-5.0, 3.0]], [[1.0, -2.0]]], np.float32),
    )
    for dtype in (ms.bfloat16, ms.float16, ms.float32):
        for func, values in ((addmm, addmm_inputs), (baddbmm, baddbmm_inputs)):
            product = np.matmul(values[1], values[2])
            inputs = tuple(ms.Tensor(value, dtype=dtype) for value in values)
            for beta in (-4.775000095367432, 0.0):
                output = run_op(func, (*inputs, beta, 2.0315001010894775), mode)
                expected = np.float32(beta) * np.broadcast_to(values[0], product.shape) + \
                    np.float32(2.0315001010894775) * product
                assert output.shape == expected.shape
                assert output.dtype == dtype
                tolerance = 1e-6 if dtype == ms.float32 else 8e-3
                np.testing.assert_allclose(output.float().asnumpy(), expected, rtol=tolerance, atol=tolerance)


def run_mode(mode, cube_math_type):
    """Check Addmm and Baddbmm in one execution mode."""
    ones_2d = np.ones((16, 16), np.float32)
    identity_2d = np.eye(16, dtype=np.float32)
    ones_3d = np.ones((3, 16, 16), np.float32)
    identity_3d = np.tile(identity_2d[None, :, :], (3, 1, 1))

    if cube_math_type != 4:
        addmm_inputs = tuple(ms.Tensor(value) for value in (ones_2d, identity_2d, identity_2d))
        baddbmm_inputs = tuple(ms.Tensor(value) for value in (ones_3d, identity_3d, identity_3d))
        expected_2d = ones_2d + identity_2d
        expected_3d = ones_3d + identity_3d
        if mode == "pynative":
            ms.set_context(mode=ms.PYNATIVE_MODE)
            addmm_output = addmm(*addmm_inputs, 1.0, 1.0)
            baddbmm_output = baddbmm(*baddbmm_inputs, 1.0, 1.0)
        else:
            addmm_output = jit(addmm, backend="ms_backend", jit_level="O0")(*addmm_inputs, 1.0, 1.0)
            baddbmm_output = jit(baddbmm, backend="ms_backend", jit_level="O0")(*baddbmm_inputs, 1.0, 1.0)
        assert addmm_output.dtype == ms.float32
        assert baddbmm_output.dtype == ms.float32
        np.testing.assert_array_equal(addmm_output.asnumpy(), expected_2d)
        np.testing.assert_array_equal(baddbmm_output.asnumpy(), expected_3d)
        if cube_math_type == 0:
            run_addmm_mode0_scalar(mode)
        return

    expected_2d = np.full((16, 16), 3.4375, np.float32)
    np.fill_diagonal(expected_2d, -1.328125)
    addmm_inputs = (
        ms.Tensor(ones_2d, dtype=ms.bfloat16),
        ms.Tensor(identity_2d, dtype=ms.bfloat16),
        ms.Tensor(identity_2d, dtype=ms.bfloat16),
    )

    expected_3d = np.tile(expected_2d[None, :, :], (3, 1, 1))
    baddbmm_inputs = (
        ms.Tensor(ones_3d, dtype=ms.bfloat16),
        ms.Tensor(identity_3d, dtype=ms.bfloat16),
        ms.Tensor(identity_3d, dtype=ms.bfloat16),
    )

    expected_fp16_2d = np.full((16, 16), np.float16(np.float32(3.4445)), np.float16)
    np.fill_diagonal(expected_fp16_2d, np.float16(np.float32(3.4445) + np.float32(-4.775)))
    expected_fp16_3d = np.tile(expected_fp16_2d[None, :, :], (3, 1, 1))
    addmm_fp16_inputs = tuple(ms.Tensor(value, dtype=ms.float16) for value in (ones_2d, identity_2d, identity_2d))
    baddbmm_fp16_inputs = tuple(
        ms.Tensor(value, dtype=ms.float16) for value in (ones_3d, identity_3d, identity_3d))

    if mode == "pynative":
        ms.set_context(mode=ms.PYNATIVE_MODE)
        addmm_output = addmm(*addmm_inputs, 3.4445, -4.775)
        baddbmm_output = baddbmm(*baddbmm_inputs, 3.4445, -4.775)
        addmm_fp16_output = addmm(*addmm_fp16_inputs, 3.4445, -4.775)
        baddbmm_fp16_output = baddbmm(*baddbmm_fp16_inputs, 3.4445, -4.775)
    else:
        addmm_output = jit(addmm, backend="ms_backend", jit_level="O0")(
            *addmm_inputs, 3.4445, -4.775)
        baddbmm_output = jit(baddbmm, backend="ms_backend", jit_level="O0")(
            *baddbmm_inputs, 3.4445, -4.775)
        addmm_fp16_output = jit(addmm, backend="ms_backend", jit_level="O0")(
            *addmm_fp16_inputs, 3.4445, -4.775)
        baddbmm_fp16_output = jit(baddbmm, backend="ms_backend", jit_level="O0")(
            *baddbmm_fp16_inputs, 3.4445, -4.775)

    assert addmm_output.dtype == ms.bfloat16
    assert baddbmm_output.dtype == ms.bfloat16
    assert addmm_fp16_output.dtype == ms.float16
    assert baddbmm_fp16_output.dtype == ms.float16
    np.testing.assert_array_equal(addmm_output.float().asnumpy(), expected_2d)
    np.testing.assert_array_equal(baddbmm_output.float().asnumpy(), expected_3d)
    np.testing.assert_array_equal(addmm_fp16_output.asnumpy(), expected_fp16_2d)
    np.testing.assert_array_equal(baddbmm_fp16_output.asnumpy(), expected_fp16_3d)
    run_mode4_broadcast(mode)


def run_cube_math_type(cube_math_type):
    """Configure one Cube mode before creating tensors, then check PyBoost and KBK."""
    ms.set_device("Ascend")
    ms.device_context.ascend.op_precision.cube_math_type(cube_math_type)
    run_mode("pynative", cube_math_type)
    run_mode("KBK", cube_math_type)


if __name__ != "__main__":
    @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2.00)
    @pytest.mark.parametrize("cube_math_type", range(5))
    def test_add_ops_cube_math_type(cube_math_type):
        """
        Feature: Configure cube_math_type for Ascend Addmm and Baddbmm.
        Description: Run each supported mode in a fresh process through the device context API.
        Expectation: Modes 0-3 match exact FP32 results and mode 4 exactly matches rounded low-precision results.
        """
        result = subprocess.run(
            [sys.executable, __file__, str(cube_math_type)],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


if __name__ == "__main__":
    run_cube_math_type(int(sys.argv[1]))
