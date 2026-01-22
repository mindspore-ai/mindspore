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
"""Tests for mindspore.mint.mul and operator '*'."""
import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, mint
from mindspore import dtype as mstype
from mindspore.graph.api import _pynative_executor

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, allclose_nparray
from tests.st.ops.test_tools.test_op import TEST_OP


def _np_rand(shape, dtype=np.float32):
    if dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8):
        return np.random.randint(1, 10, size=shape).astype(dtype)
    if dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(dtype)
    if dtype in (np.complex64, np.complex128):
        return (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
    return np.random.randn(*shape).astype(dtype)


def mul_forward_func(input_x, other):
    return mint.mul(input_x, other)


class MulNet(ms.nn.Cell):
    def construct(self, input_x, other):
        return mint.mul(input_x, other)


class MulOperatorNet(ms.nn.Cell):
    def construct(self, input_x, other):
        return input_x * other


class TestMulModule:
    def __init__(self, x_np, y_np, ms_type_x, ms_type_y):
        self.x_np = x_np
        self.y_np = y_np
        self.ms_type_x = ms_type_x
        self.ms_type_y = ms_type_y
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

        net = MulOperatorNet() if use_operator else MulNet()
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

        out = torch.mul(input_x, input_y)
        if out.dtype == torch.bfloat16:
            return out.float().numpy()
        return out.numpy()

    def forward_cmp(self):
        # Compare mint.mul
        out_ms = self.forward_mindspore_impl(use_operator=False)
        out_th = self.forward_torch_impl()

        out_ms_np = out_ms.float().asnumpy() if out_ms.dtype == mstype.bfloat16 else out_ms.asnumpy()
        allclose_nparray(out_th, out_ms_np, self.loss, self.loss)

        # Compare operator '*'
        out_ms_op = self.forward_mindspore_impl(use_operator=True)
        out_ms_op_np = out_ms_op.float().asnumpy() if out_ms_op.dtype == mstype.bfloat16 else out_ms_op.asnumpy()
        allclose_nparray(out_th, out_ms_op_np, self.loss, self.loss)

    def grad_mindspore_impl(self, use_operator=False):
        input_x = Tensor(self.x_np, self.ms_type_x)
        if isinstance(self.y_np, (int, float, bool)):
            input_y = self.y_np
        else:
            input_y = Tensor(self.y_np, self.ms_type_y)

        net = MulOperatorNet() if use_operator else MulNet()
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()

        # Generate random sensitivity
        # Fix: sens dtype should match output dtype
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
        return grads

    def grad_torch_impl(self):
        input_x = torch.from_numpy(self.x_np)
        if self.ms_type_x == mstype.bfloat16:
            input_x = input_x.bfloat16()

        # Fix: only require grad for floating/complex types
        if input_x.is_floating_point() or input_x.is_complex():
            input_x.requires_grad = True

        if isinstance(self.y_np, (int, float, bool)):
            input_y = self.y_np
            is_scalar = True
        else:
            input_y = torch.from_numpy(self.y_np)
            if self.ms_type_y == mstype.bfloat16:
                input_y = input_y.bfloat16()
            # Fix: only require grad for floating/complex types
            if input_y.is_floating_point() or input_y.is_complex():
                input_y.requires_grad = True
            is_scalar = False

        out = torch.mul(input_x, input_y)

        # Convert sens to matching torch dtype
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
        # Test grad for function mint.mul
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

        # Test grad for operator '*'
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
    mstype.bool_: np.bool_
}

def is_float_type(dtype):
    return dtype in (mstype.float16, mstype.float32, mstype.float64, mstype.bfloat16)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype_x', DTYPE_MAP.keys())
@pytest.mark.parametrize('dtype_y', DTYPE_MAP.keys())
def test_mint_mul_mixed_precision_combinations(context_mode, dtype_x, dtype_y):
    """
    Feature: mint.mul mixed precision
    Description: Test various combinations of types
    Expectation: Success
    """
    _set_mode(context_mode)
    shape = (16, 16)

    x = _np_rand(shape, DTYPE_MAP[dtype_x])
    y = _np_rand(shape, DTYPE_MAP[dtype_y])

    module = TestMulModule(x, y, dtype_x, dtype_y)
    module.forward_cmp()

    # Verify backward only if both inputs are floating point types or at least one is float (type promotion)
    if is_float_type(dtype_x) or is_float_type(dtype_y):
        module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('shapes', [
    ((3, 1, 5), (1, 4, 5)), # Basic broadcast
    ((1, 64, 1, 1), (32, 1, 1, 1)), # Multi-dim broadcast
    ((1,), (10, 10)), # Scalar-like tensor broadcast
    ((5, 1, 3), (1, 4, 1)), # Interleaved broadcast
])
def test_mint_mul_broadcast(context_mode, shapes):
    """
    Feature: mint.mul
    Description: Test broadcasting shapes
    Expectation: Success
    """
    _set_mode(context_mode)
    x = _np_rand(shapes[0], np.float32)
    y = _np_rand(shapes[1], np.float32)

    module = TestMulModule(x, y, mstype.float32, mstype.float32)
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('params', [
    # 7D: (2, 1, 2, 1, 4, 8, 8) * (2, 3, 2, 5, 4, 1, 1) -> (2, 3, 2, 5, 4, 8, 8), FP32
    {'shape1': (2, 1, 2, 1, 4, 8, 8), 'shape2': (2, 3, 2, 5, 4, 1, 1), 'dtype': mstype.float32},
    # 4D: Large Shape, Same shape, BF16
    {'shape1': (16, 200, 400, 15), 'shape2': (16, 200, 400, 15), 'dtype': mstype.float16},
    # 6D: Large Shape, Same shape, FP32
    {'shape1': (8, 8, 8, 8, 8, 8), 'shape2': (8, 8, 8, 8, 8, 8), 'dtype': mstype.float32},
    # 8D: Same shape, FP16
    {'shape1': (2, 2, 2, 2, 2, 2, 2, 2), 'shape2': (2, 2, 2, 2, 2, 2, 2, 2), 'dtype': mstype.float16},
])
def test_mint_mul_large_shape_high_dims(context_mode, params):
    """
    Feature: mint.mul large shape
    Description: Test large shapes and high dimensions (up to 8D)
    Expectation: Success
    """
    _set_mode(context_mode)
    dtype = params['dtype']
    np_dtype = DTYPE_MAP[dtype]

    x = _np_rand(params['shape1'], np_dtype)
    y = _np_rand(params['shape2'], np_dtype)

    module = TestMulModule(x, y, dtype, dtype)
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype_x', DTYPE_MAP.keys())
@pytest.mark.parametrize('scalar_y', [2.0, 2, True])
def test_mint_mul_scalar_tensor_promotion(context_mode, dtype_x, scalar_y):
    """
    Feature: mint.mul type promotion with scalars
    Description: Verify scalar does not promote tensor type if tensor type level is sufficient
    Expectation: Success
    """
    _set_mode(context_mode)
    x = _np_rand((5,), DTYPE_MAP[dtype_x])
    y = scalar_y

    module = TestMulModule(x, y, dtype_x, None)
    module.forward_cmp()

    if is_float_type(dtype_x):
        module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
def test_mint_mul_special_values(context_mode):
    """
    Feature: mint.mul robustness
    Description: Test inf and nan
    Expectation: Success
    """
    _set_mode(context_mode)
    x = np.array([np.inf, np.nan, 0.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 1.0, np.inf, np.nan], dtype=np.float32)

    module = TestMulModule(x, y, mstype.float32, mstype.float32)
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
def test_mint_mul_empty_tensor(context_mode):
    """
    Feature: mint.mul empty tensor
    Description: Test with empty shape using np.empty
    Expectation: Success
    """
    _set_mode(context_mode)
    # Generate empty tensor with specific shape (0, 8)
    x = np.empty((0, 8), dtype=np.float32)
    y = np.empty((0, 8), dtype=np.float32)

    module = TestMulModule(x, y, mstype.float32, mstype.float32)
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk', 'ge'])
def test_mint_mul_exception_shape_mismatch(context_mode):
    """
    Feature: mint.mul exception
    Description: Test shape mismatch that cannot broadcast
    Expectation: ValueError
    """
    _set_mode(context_mode)
    x = Tensor(np.ones((3, 4)), mstype.float32)
    y = Tensor(np.ones((2, 5)), mstype.float32)

    with pytest.raises(ValueError):
        mint.mul(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', ['pynative', 'kbk'])
def test_mint_mul_dynamic_shape_test_op(context_mode):
    """
    Feature: Dynamic shape/rank for mint.mul
    Description: use TEST_OP to validate dynamic rank/shape
    Expectation: Success
    """
    _set_mode(context_mode)
    x1 = Tensor(_np_rand((10, 10, 8), np.float32), mstype.float32)
    y1 = Tensor(_np_rand((10, 10, 8), np.float32), mstype.float32)

    x2 = Tensor(_np_rand((5, 10), np.float32), mstype.float32)
    y2 = Tensor(_np_rand((5, 10), np.float32), mstype.float32)

    TEST_OP(mul_forward_func,
            [[x1, y1], [x2, y2]],
            disable_case=['EmptyTensor'],
            disable_mode=["GRAPH_MODE_GE"])
