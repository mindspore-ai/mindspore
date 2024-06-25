# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test Primitive's arguments dtype auto-cast with one stage"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import ops
from mindspore.common.api import jit
from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "compile_by_trace": True,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_add_fp16_and_fp32():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.add(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3], dtype=ms.float16)
    b = Tensor([1, 2, 3], dtype=ms.float32)
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([2, 4, 6], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_add_fp32_and_int32():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.add(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3], dtype=ms.float32)
    b = Tensor([1, 2, 3], dtype=ms.int32)
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([2, 4, 6], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_add_fp32_and_int_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.add(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3], dtype=ms.float32)
    b = 2
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([3, 4, 5], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_add_fp16_and_int_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.add(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3], dtype=ms.float16)
    b = 2
    ret = fn(a, b)

    assert ret.dtype == ms.float16
    match_array(ret.asnumpy(), Tensor([3, 4, 5], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mul_fp16_and_fp32():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.mul(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float16)
    b = Tensor([1., 2., 3.], dtype=ms.float32)
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([1, 4, 9], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mul_fp32_and_int32():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.mul(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float32)
    b = Tensor([1, 2, 3], dtype=ms.int32)
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([1, 4, 9], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mul_fp32_and_int_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.mul(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3], dtype=ms.float32)
    b = 2
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([2, 4, 6], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mul_fp16_and_int_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.mul(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float16)
    b = 2
    ret = fn(a, b)

    assert ret.dtype == ms.float16
    match_array(ret.asnumpy(), Tensor([2, 4, 6], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pow_fp16_and_fp32():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.pow(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float16)
    b = Tensor([1., 2., 3.], dtype=ms.float32)
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([1, 4, 27], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pow_fp32_and_int_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.pow(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float32)
    b = 2
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([1, 4, 9], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_div_fp16_and_fp32():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.div(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float16)
    b = Tensor([4., 5., 6.], dtype=ms.float32)
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([0.25, 0.4, 0.5], dtype=ms.float32).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_div_fp32_and_int_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.div(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1., 2., 3.], dtype=ms.float32)
    b = 2
    ret = fn(a, b)

    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([0.5, 1., 1.5], dtype=ms.float32).asnumpy())


################# auto-cast by arg_handler ######################


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zeros_dtype_arg_handler_dtype_to_type_id():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(shape, dtype):
        return ops.zeros(shape, dtype)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = fn((2, 3), ms.float16)

    assert ret.dtype == ms.float16
    match_array(ret.asnumpy(), np.zeros((2, 3), np.float16))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_div_rounding_mode_arg_handler_str_to_enum():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y, mode):
        return ops.div(x, y, rounding_mode=mode)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([4., -5., 3.], dtype=ms.float32)
    b = Tensor([1.5, 2., 6.], dtype=ms.float32)

    ret = fn(a, b, 'floor')
    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([2., -3., 0.], dtype=ms.float32).asnumpy())

    ret = fn(a, b, 'trunc')
    assert ret.dtype == ms.float32
    match_array(ret.asnumpy(), Tensor([2., -2., 0.], dtype=ms.float32).asnumpy())


################# auto-cast by type_cast ######################


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_less_type_cast_from_number_bool_to_tensor():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        return ops.less(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = fn(True, 5)
    assert ret == True

    ret = fn(False, -1)
    assert ret == False


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reshape_type_cast_from_list_tensor_to_tuple():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, shape):
        return ops.reshape(x, shape)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3, 4, 5, 6])
    shape = [2, 3]
    ret = fn(a, shape)
    match_array(ret.asnumpy(), np.array([[1, 2, 3], [4, 5, 6]]))

    a = Tensor([1, 2, 3, 4, 5, 6])
    shape = Tensor([2, 3])
    ret = fn(a, shape)
    match_array(ret.asnumpy(), np.array([[1, 2, 3], [4, 5, 6]]))


################# args with default-value ######################


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_concat_axis_default_value():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(tensors, axis=None):
        return ops.cat(tensors, axis=axis) if axis is not None else ops.cat(tensors)

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1]]))
    x2 = Tensor(np.array([[0, 1], [2, 1]]))
    ret = fn((x1, x2))
    match_array(ret.asnumpy(), np.array([[0, 1], [2, 1], [0, 1], [2, 1]]))

    x1 = Tensor(np.array([[0, 1], [2, 1]]))
    x2 = Tensor(np.array([[0, 1], [2, 1]]))
    ret = fn((x1, x2), axis=1)
    match_array(ret.asnumpy(), np.array([[0, 1, 0, 1], [2, 1, 2, 1]]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pad_mode_default_value():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, padding, mode=None):
        return ops.pad(x, padding, mode=mode) if mode is not None else ops.pad(x, padding)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.arange(6).reshape((2, 3)))
    pad = [1, 1]
    ret = fn(x, pad)
    match_array(ret.asnumpy(), np.array([[0, 0, 1, 2, 0], [0, 3, 4, 5, 0]]))

    x = Tensor(np.arange(6).reshape((2, 3)))
    pad = [1, 1]
    ret = fn(x, pad, mode='replicate')
    match_array(ret.asnumpy(), np.array([[0, 0, 1, 2, 2], [3, 3, 4, 5, 5]]))


################# primitive with __init__() args ######################


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_MutMul_with_no_init_args():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        op = ops.MatMul()
        return op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 2, 3], [1, 2, 3]], dtype=ms.float32)
    b = Tensor([[1], [2], [3]], dtype=ms.float32)
    ret = fn(a, b)

    match_array(ret.asnumpy(), np.array([[14], [14]]).astype(np.float32))
    assert_executed_by_graph_mode(fn)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_MutMul_with_two_init_args():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y, transpose_a=None, transpose_b=None):
        op = ops.MatMul(transpose_a=transpose_a, transpose_b=transpose_b)
        return op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 1], [2, 2], [3, 3]], dtype=ms.float32)
    b = Tensor([[1, 2, 3]], dtype=ms.float32)
    ret = fn(a, b, transpose_a=True, transpose_b=True)

    match_array(ret.asnumpy(), np.array([[14], [14]]).astype(np.float32))
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_MutMul_with_one_init_arg_transpose_a():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y, transpose_a=None):
        op = ops.MatMul(transpose_a=transpose_a)
        return op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 1], [2, 2], [3, 3]], dtype=ms.float32)
    b = Tensor([[1], [2], [3]], dtype=ms.float32)
    ret = fn(a, b, transpose_a=True)

    match_array(ret.asnumpy(), np.array([[14], [14]]).astype(np.float32))
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_MutMul_with_one_init_arg_transpose_b():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y, transpose_b=None):
        op = ops.MatMul(transpose_b=transpose_b)
        return op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 2, 3], [1, 2, 3]], dtype=ms.float32)
    b = Tensor([[1, 2, 3]], dtype=ms.float32)
    ret = fn(a, b, transpose_b=True)

    match_array(ret.asnumpy(), np.array([[14], [14]]).astype(np.float32))
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_BiasAdd_with_no_init_args():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        op = ops.BiasAdd()
        return op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 2, 3], [2, 3, 4]], dtype=ms.float32)
    b = Tensor([1, 1, 1], dtype=ms.float32)
    ret = fn(a, b)

    match_array(ret.asnumpy(), np.array([[2, 3, 4], [3, 4, 5]]).astype(np.float32))
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_BiasAdd_with_one_init_args():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x, y):
        op = ops.BiasAdd(data_format='NCHW')
        return op(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([[1, 2, 3], [2, 3, 4]], dtype=ms.float32)
    b = Tensor([1, 1, 1], dtype=ms.float32)
    ret = fn(a, b)

    match_array(ret.asnumpy(), np.array([[2, 3, 4], [3, 4, 5]]).astype(np.float32))
    assert_executed_by_graph_mode(fn)
