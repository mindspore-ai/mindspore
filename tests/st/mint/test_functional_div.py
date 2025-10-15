# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Tests for mindspore.mint.div and operator '/'."""
import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, mint
from mindspore import dtype as mstype
from mindspore.common.api import _pynative_executor

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import allclose_nparray
from tests.st.ops.test_tools.test_op import TEST_OP


def _np_rand(shape, dtype=np.float32):
    if dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8):
        return np.random.randint(1, 10, size=shape).astype(dtype)
    if dtype == np.bool_:
        return np.ones(shape, dtype=dtype)
    if dtype in (np.complex64, np.complex128):
        # Avoid 0 in denominator
        data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        data[np.abs(data) < 0.1] = 0.1 + 0.1j
        return data
    # Avoid small values close to 0 for floats to prevent instability
    data = np.random.randn(*shape).astype(dtype)
    data[np.abs(data) < 0.1] = 0.1
    return data


class DivNet(ms.nn.Cell):
    def __init__(self, rounding_mode=None):
        super().__init__()
        self.rounding_mode = rounding_mode

    def construct(self, input_x, other):
        return mint.div(input_x, other, rounding_mode=self.rounding_mode)


class DivOperatorNet(ms.nn.Cell):
    def construct(self, input_x, other):
        return input_x / other


class TestDivModule:
    def __init__(self, x_np, y_np, ms_type_x, ms_type_y, rounding_mode=None):
        self.x_np = x_np
        self.y_np = y_np
        self.ms_type_x = ms_type_x
        self.ms_type_y = ms_type_y
        self.rounding_mode = rounding_mode
        self.loss = 1e-4
        if ms_type_x == mstype.float16 or ms_type_y == mstype.float16:
            self.loss = 1e-3
        if ms_type_x == mstype.float64 or ms_type_y == mstype.float64:
            self.loss = 1e-5
        if ms_type_x == mstype.bfloat16 or ms_type_y == mstype.bfloat16:
            self.loss = 4e-3

    def forward_mindspore_impl(self, use_operator=False):
        input_x = Tensor(self.x_np, self.ms_type_x)
        if isinstance(self.y_np, (int, float, bool)):
            input_y = self.y_np
        else:
            input_y = Tensor(self.y_np, self.ms_type_y)

        net = DivOperatorNet() if use_operator else DivNet(self.rounding_mode)
        out = net(input_x, input_y)
        return out

    def forward_torch_impl(self):
        input_x = torch.from_numpy(self.x_np)
        if self.ms_type_x == mstype.bfloat16:
            input_x = input_x.bfloat16()

        if isinstance(self.y_np, (int, float, bool)):
            input_y = self.y_np
        else:
            input_y = torch.from_numpy(self.y_np)
            if self.ms_type_y == mstype.bfloat16:
                input_y = input_y.bfloat16()

        out = torch.div(input_x, input_y, rounding_mode=self.rounding_mode)
        if out.dtype == torch.bfloat16:
            return out.float().numpy()
        return out.numpy()

    def forward_cmp(self):
        # Compare mint.div
        out_ms = self.forward_mindspore_impl(use_operator=False)
        out_th = self.forward_torch_impl()

        out_ms_np = out_ms.float().asnumpy() if out_ms.dtype == mstype.bfloat16 else out_ms.asnumpy()
        allclose_nparray(out_th, out_ms_np, self.loss, self.loss)

        # Compare operator '/'
        # Only check operator '/' when rounding_mode is None
        if self.rounding_mode is None:
            out_ms_op = self.forward_mindspore_impl(use_operator=True)
            out_ms_op_np = out_ms_op.float().asnumpy() if out_ms_op.dtype == mstype.bfloat16 else out_ms_op.asnumpy()
            allclose_nparray(out_th, out_ms_op_np, self.loss, self.loss)

    def grad_mindspore_impl(self, use_operator=False):
        input_x = Tensor(self.x_np, self.ms_type_x)
        if isinstance(self.y_np, (int, float, bool)):
            input_y = self.y_np
        else:
            input_y = Tensor(self.y_np, self.ms_type_y)
        grad_position = []
        if is_float_type(self.ms_type_x):
            grad_position.append(0)
        if is_float_type(self.ms_type_y):
            grad_position.append(1)
        net = DivOperatorNet() if use_operator else DivNet(self.rounding_mode)
        grad_net = ms.ops.GradOperation(get_by_position=True, sens_param=True)(net, grad_position=tuple(grad_position))

        out = self.forward_mindspore_impl(use_operator)
        out_shape = out.shape
        out_dtype = out.dtype

        # Generate float32 random data first
        sens_np = _np_rand(out_shape, dtype=np.float32)
        sens = Tensor(sens_np, dtype=mstype.float32).astype(out_dtype)

        if isinstance(input_y, (int, float, bool)):
            grads = grad_net(input_x, input_y, sens)
        else:
            grads = grad_net(input_x, input_y, sens)

        self.sens_np = sens.float().asnumpy() if sens.dtype == mstype.bfloat16 else sens.asnumpy()
        grad_x = None
        grad_y = None
        if is_float_type(self.ms_type_x) and len(grad_position) == 1:
            grad_x = grads
        elif is_float_type(self.ms_type_y) and len(grad_position) == 1:
            grad_y = grads
        elif is_float_type(self.ms_type_x) and is_float_type(self.ms_type_y):
            grad_x = grads[0]
            grad_y = grads[1]
        return (grad_x, grad_y)

    def grad_torch_impl(self):
        input_x = torch.from_numpy(self.x_np)
        if self.ms_type_x == mstype.bfloat16:
            input_x = input_x.bfloat16()

        if input_x.is_floating_point() or input_x.is_complex():
            input_x.requires_grad = True

        if isinstance(self.y_np, (int, float, bool)):
            input_y = self.y_np
            is_scalar = True
        else:
            input_y = torch.from_numpy(self.y_np)
            if self.ms_type_y == mstype.bfloat16:
                input_y = input_y.bfloat16()
            if input_y.is_floating_point() or input_y.is_complex():
                input_y.requires_grad = True
            is_scalar = False

        out = torch.div(input_x, input_y, rounding_mode=self.rounding_mode)

        sens_np = self.sens_np
        if out.dtype == torch.bfloat16:
            sens = torch.from_numpy(sens_np).bfloat16()
        elif out.dtype == torch.float16:
            sens = torch.from_numpy(sens_np).half()
        elif out.dtype == torch.float64:
            sens = torch.from_numpy(sens_np).double()
        else:
            sens = torch.from_numpy(sens_np).float()

        out.backward(sens)

        gx = None
        if input_x.grad is not None:
            gx = input_x.grad.float().numpy() if input_x.dtype == torch.bfloat16 else input_x.grad.numpy()

        if is_scalar:
            return (gx,)

        gy = None
        if input_y.grad is not None:
            gy = input_y.grad.float().numpy() if input_y.dtype == torch.bfloat16 else input_y.grad.numpy()

        return (gx, gy)

    def grad_cmp(self):
        # Test grad for function mint.div
        grads_ms = self.grad_mindspore_impl(use_operator=False)
        grads_th = self.grad_torch_impl()

        # Compare first input grad
        if grads_th[0] is not None:
            g_ms_np = grads_ms[0].float().asnumpy() if grads_ms[0].dtype == mstype.bfloat16 else grads_ms[0].asnumpy()
            allclose_nparray(grads_th[0], g_ms_np, self.loss, self.loss)

        # Compare second input grad if exists
        if len(grads_th) > 1 and grads_th[1] is not None:
            g_ms_np = grads_ms[1].float().asnumpy() if grads_ms[1].dtype == mstype.bfloat16 else grads_ms[1].asnumpy()
            allclose_nparray(grads_th[1], g_ms_np, self.loss, self.loss)

        # Test grad for operator '/'
        if self.rounding_mode is None:
            grads_ms_op = self.grad_mindspore_impl(use_operator=True)
            grads_th = self.grad_torch_impl()
            # Re-use torch grad as baseline

            # Compare first input grad
            if grads_th[0] is not None:
                g_ms_op_np = (grads_ms_op[0].float().asnumpy() if grads_ms_op[0].dtype == mstype.bfloat16
                              else grads_ms_op[0].asnumpy())
                allclose_nparray(grads_th[0], g_ms_op_np, self.loss, self.loss)

            # Compare second input grad if exists
            if len(grads_th) > 1 and grads_th[1] is not None:
                g_ms_op_np = (grads_ms_op[1].float().asnumpy() if grads_ms_op[1].dtype == mstype.bfloat16
                              else grads_ms_op[1].asnumpy())
                allclose_nparray(grads_th[1], g_ms_op_np, self.loss, self.loss)


def _set_mode(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')


DTYPE_MAP = {
    mstype.float16: np.float16,
    mstype.float32: np.float32,
    mstype.float64: np.float64,
    mstype.bfloat16: np.float32, # numpy no bf16, use fp32
    mstype.int8: np.int8,
    mstype.int16: np.int16,
    mstype.int32: np.int32,
    mstype.int64: np.int64,
    mstype.uint8: np.uint8,
}

def is_float_type(dtype):
    return dtype in (mstype.float16, mstype.float32, mstype.float64, mstype.bfloat16)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype_x', DTYPE_MAP.keys())
@pytest.mark.parametrize('dtype_y', DTYPE_MAP.keys())
@pytest.mark.parametrize('rounding_mode', [None])
def test_mint_div_mixed_precision_combinations(context_mode, dtype_x, dtype_y, rounding_mode):
    """
    Feature: mint.div mixed precision
    Description: Test various combinations of types
    Expectation: Success
    """
    _set_mode(context_mode)
    shape = (16, 16)

    x = _np_rand(shape, DTYPE_MAP[dtype_x])
    y = _np_rand(shape, DTYPE_MAP[dtype_y])

    module = TestDivModule(x, y, dtype_x, dtype_y, rounding_mode)
    module.forward_cmp()

    # Verify backward only if both inputs are floating point types or at least one is float (type promotion)
    if is_float_type(dtype_x) or is_float_type(dtype_y):
        module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1', card_mark='onecard', mem_peak=5.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('rounding_mode', ['floor', 'trunc'])
def test_mint_div_mixed_precision_combinations_fixed_inputs(context_mode, rounding_mode):
    """
    Feature: mint.div mixed precision
    Description: Use fixed inputs to avoid flaky off-by-one results around integer boundaries under floor/trunc.
    Expectation: Success
    """
    _set_mode(context_mode)

    cases = [
        # float32 + float16
        (mstype.float32, mstype.float16,
         np.array([10.5, -10.5, 7.0, -7.0, 1.25, -1.25], dtype=np.float32),
         np.array([2.0, 2.0, -3.0, -3.0, 0.5, 0.5], dtype=np.float16)),
        # float32 + float64
        (mstype.float32, mstype.float64,
         np.array([11.0, -11.0, 7.0, -7.0, 1.0, -1.0], dtype=np.float32),
         np.array([2.5, 2.5, -3.0, -3.0, 0.3, 0.3], dtype=np.float64)),
        # float32 + int32
        (mstype.float32, mstype.int32,
         np.array([9.0, -9.0, 10.0, -10.0, 7.0, -7.0], dtype=np.float32),
         np.array([2, 2, -4, -4, 3, 3], dtype=np.int32)),
        # float16 + int32
        (mstype.float16, mstype.int32,
         np.array([9.0, -9.0, 10.0, -10.0, 7.0, -7.0], dtype=np.float16),
         np.array([2, 2, -4, -4, 3, 3], dtype=np.int32)),
    ]

    for dtype_x, dtype_y, x, y in cases:
        # Some GE kernels (e.g. Trunc/Floor) may not support float64.
        if context_mode == 'ge' and (dtype_x == mstype.float64 or dtype_y == mstype.float64):
            continue

        module = TestDivModule(x, y, dtype_x, dtype_y, rounding_mode)
        module.forward_cmp()
        if is_float_type(dtype_x) or is_float_type(dtype_y):
            module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('shapes', [
    ((3, 1, 5), (1, 4, 5)), # Basic broadcast
    ((1, 64, 1, 1), (32, 1, 1, 1)), # Multi-dim broadcast
    ((1,), (10, 10)), # Scalar-like tensor broadcast
    ((5, 1, 3), (1, 4, 1)), # Interleaved broadcast
])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_mint_div_broadcast(context_mode, shapes, rounding_mode):
    """
    Feature: mint.div
    Description: Test broadcasting shapes
    Expectation: Success
    """
    _set_mode(context_mode)
    x = _np_rand(shapes[0], np.float32)
    y = _np_rand(shapes[1], np.float32)

    module = TestDivModule(x, y, mstype.float32, mstype.float32, rounding_mode)
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype_x', DTYPE_MAP.keys())
@pytest.mark.parametrize('scalar_y', [2.0, 2])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_mint_div_scalar_tensor_promotion(context_mode, dtype_x, scalar_y, rounding_mode):
    """
    Feature: mint.div type promotion with scalars
    Description: Verify scalar does not promote tensor type if tensor type level is sufficient
    Expectation: Success
    """
    if context_mode == 'ge' and dtype_x == mstype.float64 and rounding_mode == 'trunc':
        pytest.skip("Op 'Trunc' does not support float64 type on GE mode.")
    _set_mode(context_mode)
    x = _np_rand((5,), DTYPE_MAP[dtype_x])
    y = scalar_y

    module = TestDivModule(x, y, dtype_x, None, rounding_mode)
    module.forward_cmp()

    if is_float_type(dtype_x):
        module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_mint_div_special_values(context_mode, rounding_mode):
    """
    Feature: mint.div robustness
    Description: Test inf, nan, and zero division
    Expectation: Success
    """
    _set_mode(context_mode)
    # x: [inf, nan, 0.0, 1.0, 1.0]
    # y: [0.0, 1.0, inf, nan, 0.0]
    # Expected: [inf/nan, nan, 0.0, nan, inf] - torch vs mindspore check
    x = np.array([np.inf, np.nan, 0.0, 1.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 1.0, np.inf, np.nan, 0.0], dtype=np.float32)

    module = TestDivModule(x, y, mstype.float32, mstype.float32, rounding_mode)
    module.forward_cmp()
    # Gradient might be unstable with NaNs/Infs, checking only forward for special values or standard tolerance
    module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_mint_div_empty_tensor(context_mode, rounding_mode):
    """
    Feature: mint.div empty tensor
    Description: Test with empty shape using np.empty
    Expectation: Success
    """
    _set_mode(context_mode)
    x = np.empty((0, 8), dtype=np.float32)
    y = np.empty((0, 8), dtype=np.float32)

    module = TestDivModule(x, y, mstype.float32, mstype.float32, rounding_mode)
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_mint_div_exception_shape_mismatch(context_mode, rounding_mode):
    """
    Feature: mint.div exception
    Description: Test shape mismatch that cannot broadcast
    Expectation: ValueError
    """
    _set_mode(context_mode)
    x = Tensor(np.ones((3, 4)), mstype.float32)
    y = Tensor(np.ones((2, 5)), mstype.float32)

    with pytest.raises(ValueError):
        mint.div(x, y, rounding_mode=rounding_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'],
          level_mark='level0', card_mark='onecard', mem_peak=68.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_mint_div_dynamic_shape_test_op(context_mode, rounding_mode):
    """
    Feature: Dynamic shape/rank for mint.div
    Description: use TEST_OP to validate dynamic rank/shape
    Expectation: Success
    """
    _set_mode(context_mode)
    # Avoid 0 in y for TEST_OP inputs
    x1_np = _np_rand((10, 10, 8), np.float32)
    y1_np = _np_rand((10, 10, 8), np.float32)
    x1 = Tensor(x1_np, mstype.float32)
    y1 = Tensor(y1_np, mstype.float32)

    x2_np = _np_rand((5, 10), np.float32)
    y2_np = _np_rand((5, 10), np.float32)
    x2 = Tensor(x2_np, mstype.float32)
    y2 = Tensor(y2_np, mstype.float32)

    def div_forward_func(input_x, other):
        return mint.div(input_x, other, rounding_mode=rounding_mode)

    TEST_OP(div_forward_func,
            [[x1, y1], [x2, y2]],
            disable_case=['EmptyTensor'],
            disable_mode=["GRAPH_MODE_GE"])
