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

"""Testcases for Tensor fallback"""

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint, ops
from mindspore._c_expression import NoFallbackGuard
from tests.mark_utils import arg_mark

class Layout():
    pass

class DTensor(Tensor):
    RUN_OPS = []

    def __new__(cls, local_tensor, layout):
        t = Tensor._make_subclass(cls, local_tensor)
        t._local_tensor = local_tensor
        t._layout = layout
        return t

    def __init__(self, local_tensor, layout):
        # Skip __init__. Tensor is created by Tensor._make_subclass in __new__.
        pass

    def asnumpy(self):
        return self._local_tensor.asnumpy()

    @classmethod
    def __fallback__(cls, func, args=(), kwargs=None):
        def wrap_dtensor(x):
            if isinstance(x, Tensor):
                return DTensor(x, Layout())
            if isinstance(x, (tuple, list)):
                return type(x)(wrap_dtensor(item) for item in x)
            return x

        DTensor.RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}

        new_args = [arg._local_tensor if isinstance(arg, DTensor) else arg for arg in args]
        with NoFallbackGuard():
            return wrap_dtensor(func(*new_args, **kwargs))


# ====== Common helper: create a DTensor from input data ======

def _make_base_dtensor(data):
    """
    Reset DTensor.RUN_OPS and create a DTensor from given data.
    """
    DTensor.RUN_OPS = []
    x = DTensor(Tensor(data), Layout())
    return x


def _make_dtensor(data):
    """Create a DTensor from given data without resetting RUN_OPS."""
    return DTensor(Tensor(data), Layout())


# ====== Scenario 1: basic add ======

def test_pynative_fallback_add():
    """
    Feature: Pynative fallback
    Description: Test dtensor add fallback.
    Expectation: Result is correct and AddExt is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert np.allclose(y.asnumpy(), np.array([2, 2, 2]))
    assert DTensor.RUN_OPS == ["AddExt"]


# ====== Scenario 2: reshape with tuple shape ======

def test_pynative_fallback_reshape_tuple():
    """
    Feature: Pynative fallback
    Description: Test dtensor reshape with tuple shape.
    Expectation: Result is correct and Reshape is recorded after AddExt.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    z = y.reshape((3, 1))
    assert np.allclose(z.asnumpy(), np.array([[2], [2], [2]]))
    assert DTensor.RUN_OPS == ["AddExt", "Reshape"]


# ====== Scenario 3: tensor overload operator (+ scalar) ======

def test_pynative_fallback_add_scalar_overload():
    """
    Feature: Pynative fallback
    Description: Test dtensor add scalar via tensor overload.
    Expectation: Result is correct and AddScalar is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    z = y.reshape((3, 1))
    w = z + 1
    assert np.allclose(w.asnumpy(), np.array([[3], [3], [3]]))
    assert DTensor.RUN_OPS == ["AddExt", "Reshape", "AddScalar"]


# ====== Scenario 4: mint.sin ======

def test_pynative_fallback_mint_sin():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.sin.
    Expectation: Result is correct and Sin is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    z = y.reshape((3, 1))
    w = z + 1
    a = mint.sin(w)
    assert np.allclose(a.asnumpy(), np.sin(np.array([[3], [3], [3]])))
    assert DTensor.RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin"]


# ====== Scenario 5: ops.Assign primitive ======

def test_pynative_fallback_assign_primitive():
    """
    Feature: Pynative fallback
    Description: Test dtensor with ops.Assign primitive.
    Expectation: Result is correct and Assign is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    z = y.reshape((3, 1))
    w = z + 1
    a = mint.sin(w)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    assert np.allclose(b.asnumpy(), expect_data)
    assert DTensor.RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign"]


# ====== Scenario 6: mint.cat ======

def test_pynative_fallback_mint_cat():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.cat.
    Expectation: Result is correct and Concat is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    z = y.reshape((3, 1))
    w = z + 1
    a = mint.sin(w)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    c = mint.cat((b, b))
    assert np.allclose(c.asnumpy(), np.array([[1], [1], [1], [1], [1], [1]]))
    assert DTensor.RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign", "Concat"]


# ====== Scenario 6b: mint.cat with list input ======

def test_pynative_fallback_mint_cat_list():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.cat using list input instead of tuple.
    Expectation: Result is correct and Concat is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    z = y.reshape((3, 1))
    w = z + 1
    a = mint.sin(w)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    c = mint.cat([b, b])
    assert np.allclose(c.asnumpy(), np.array([[1], [1], [1], [1], [1], [1]]))
    assert DTensor.RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign", "Concat"]


# ====== Scenario 7: primitive attribute Split ======

def test_pynative_fallback_split_primitive_attr():
    """
    Feature: Pynative fallback
    Description: Test dtensor with primitive attribute Split.
    Expectation: Split result shape/value is correct and Split is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)

    assert len(y) == 2
    assert np.allclose(y[0].asnumpy(), np.array([1, 1]))
    assert np.allclose(y[1].asnumpy(), np.array([1, 1]))
    assert DTensor.RUN_OPS == ["Split"]


# ====== Scenario 8: ops.cast ======

def test_pynative_fallback_cast():
    """
    Feature: Pynative fallback
    Description: Test dtensor with ops.cast.
    Expectation: Result is correct and Cast is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)
    y0 = ms.ops.cast(y[0], ms.float32)

    assert np.allclose(y0.asnumpy(), np.array([1.0, 1.0]))
    assert DTensor.RUN_OPS == ["Split", "Cast"]


# ====== Scenario 9: Tensor.to(dtype) ======

def test_pynative_fallback_to_dtype():
    """
    Feature: Pynative fallback
    Description: Test dtensor to(ms.float32).
    Expectation: Result is correct and ToDtype is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)
    y1 = y[1].to(ms.float32)

    assert np.allclose(y1.asnumpy(), np.array([1.0, 1.0]))
    assert DTensor.RUN_OPS == ["Split", "ToDtype"]


def test__update_data_shallow_copy_tensor():
    """
    Feature: Pynative fallback
    Description: Test Tensor._update_data.
    Expectation: data should be updated (shallow copy).
    """
    data1 = np.array([1, 2, 3], dtype=np.float32)
    data2 = np.array([4, 5, 6], dtype=np.float32)

    t1 = Tensor(data1)
    t2 = Tensor(data2)

    t1._update_data(t2)

    np.testing.assert_allclose(t1.asnumpy(), t2.asnumpy())
    assert t1._data_ptr() == t2._data_ptr()


def test__update_data_raise_on_non_tensor():
    """
    Feature: Pynative fallback
    Description: Test Tensor._update_data with invalid input.
    Expectation: Raise RuntimeError.
    """
    t = Tensor(np.array([1, 2, 3], dtype=np.float32))
    not_tensor = 123

    with pytest.raises(RuntimeError, match="Only support input type Tensor"):
        t._update_data(not_tensor)

# ====== Scenario 10: sub tensor-tensor (overloaded, pyboost_overload_mint_class_def.tpl) ======

def test_pynative_fallback_sub():
    """
    Feature: Pynative fallback
    Description: Test dtensor sub (tensor-tensor) fallback.
    Expectation: Result is correct and SubExt is recorded.
    """
    x = _make_base_dtensor(np.array([5, 4, 3], dtype=np.float32))
    y = _make_dtensor(np.array([1, 2, 1], dtype=np.float32))
    z = x.sub(y)
    assert np.allclose(z.asnumpy(), np.array([4, 2, 2]))
    assert DTensor.RUN_OPS == ["SubExt"]


# ====== Scenario 11: sub scalar (overloaded, scalar dispatch path) ======

def test_pynative_fallback_sub_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor subtract scalar fallback.
    Expectation: Result is correct and SubScalar is recorded.
    """
    x = _make_base_dtensor(np.array([5, 4, 3], dtype=np.float32))
    z = x - 1
    assert np.allclose(z.asnumpy(), np.array([4, 3, 2]))
    assert DTensor.RUN_OPS == ["SubScalar"]


# ====== Scenario 12: sub with alpha kwonlyarg ======

def test_pynative_fallback_sub_alpha():
    """
    Feature: Pynative fallback
    Description: Test dtensor sub with alpha keyword-only arg.
    Expectation: Result = x - alpha * y, SubExt is recorded.
    """
    x = _make_base_dtensor(np.array([10, 10, 10], dtype=np.float32))
    y = _make_dtensor(np.array([1, 2, 3], dtype=np.float32))
    z = x.sub(y, alpha=2)
    assert np.allclose(z.asnumpy(), np.array([8, 6, 4]))
    assert DTensor.RUN_OPS == ["SubExt"]


# ====== Scenario 13: sub via __sub__ operator ======

def test_pynative_fallback_operator_sub():
    """
    Feature: Pynative fallback
    Description: Test dtensor - operator (tensor-tensor) fallback via __sub__ alias.
    Expectation: Result is correct and SubExt is recorded.
    """
    x = _make_base_dtensor(np.array([5, 4, 3], dtype=np.float32))
    y = _make_dtensor(np.array([1, 2, 1], dtype=np.float32))
    z = x - y
    assert np.allclose(z.asnumpy(), np.array([4, 2, 2]))
    assert DTensor.RUN_OPS == ["SubExt"]


# ====== Scenario 14: mul tensor-tensor (overloaded) ======

def test_pynative_fallback_mul():
    """
    Feature: Pynative fallback
    Description: Test dtensor mul (tensor-tensor) fallback.
    Expectation: Result is correct and Mul is recorded.
    """
    x = _make_base_dtensor(np.array([2, 3, 4], dtype=np.float32))
    y = _make_dtensor(np.array([3, 2, 1], dtype=np.float32))
    z = x.mul(y)
    assert np.allclose(z.asnumpy(), np.array([6, 6, 4]))
    assert DTensor.RUN_OPS == ["Mul"]


# ====== Scenario 15: mul scalar ======

def test_pynative_fallback_mul_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor mul scalar fallback.
    Expectation: Result is correct and Muls is recorded.
    """
    x = _make_base_dtensor(np.array([2, 3, 4], dtype=np.float32))
    z = x * 3
    assert np.allclose(z.asnumpy(), np.array([6, 9, 12]))
    assert DTensor.RUN_OPS == ["Muls"]


# ====== Scenario 16: mul via __mul__ operator ======

def test_pynative_fallback_operator_mul():
    """
    Feature: Pynative fallback
    Description: Test dtensor * operator (tensor-tensor) fallback via __mul__ alias.
    Expectation: Result is correct and Mul is recorded.
    """
    x = _make_base_dtensor(np.array([2, 3, 4], dtype=np.float32))
    y = _make_dtensor(np.array([3, 2, 1], dtype=np.float32))
    z = x * y
    assert np.allclose(z.asnumpy(), np.array([6, 6, 4]))
    assert DTensor.RUN_OPS == ["Mul"]


# ====== Scenario 17: div tensor-tensor (overloaded) ======

def test_pynative_fallback_div():
    """
    Feature: Pynative fallback
    Description: Test dtensor div (tensor-tensor) fallback.
    Expectation: Result is correct and Div is recorded.
    """
    x = _make_base_dtensor(np.array([6, 9, 12], dtype=np.float32))
    y = _make_dtensor(np.array([2, 3, 4], dtype=np.float32))
    z = x.div(y)
    assert np.allclose(z.asnumpy(), np.array([3, 3, 3]))
    assert DTensor.RUN_OPS == ["Div"]


# ====== Scenario 18: div scalar ======

def test_pynative_fallback_div_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor div scalar fallback.
    Expectation: Result is correct and Divs is recorded.
    """
    x = _make_base_dtensor(np.array([6, 9, 12], dtype=np.float32))
    z = x / 3
    assert np.allclose(z.asnumpy(), np.array([2, 3, 4]))
    assert DTensor.RUN_OPS == ["Divs"]


# ====== Scenario 19: div via __truediv__ operator ======

def test_pynative_fallback_operator_truediv():
    """
    Feature: Pynative fallback
    Description: Test dtensor / operator (tensor-tensor) fallback via __truediv__ alias.
    Expectation: Result is correct and Div is recorded.
    """
    x = _make_base_dtensor(np.array([6, 9, 12], dtype=np.float32))
    y = _make_dtensor(np.array([2, 3, 4], dtype=np.float32))
    z = x / y
    assert np.allclose(z.asnumpy(), np.array([3, 3, 3]))
    assert DTensor.RUN_OPS == ["Div"]


# ====== Scenario 20: div with rounding_mode kwonlyarg ======

def test_pynative_fallback_div_rounding_mode():
    """
    Feature: Pynative fallback
    Description: Test dtensor div with rounding_mode keyword-only arg.
    Expectation: Result is truncated and DivMod is recorded.
    """
    x = _make_base_dtensor(np.array([7, 8, 9], dtype=np.float32))
    y = _make_dtensor(np.array([2, 3, 4], dtype=np.float32))
    z = x.div(y, rounding_mode='trunc')
    assert np.allclose(z.asnumpy(), np.array([3, 2, 2]))
    assert DTensor.RUN_OPS == ["DivMod"]


# ====== Scenario 21: add with alpha kwonlyarg ======

def test_pynative_fallback_add_alpha():
    """
    Feature: Pynative fallback
    Description: Test dtensor add with alpha keyword-only arg.
    Expectation: Result = x + alpha * y, AddExt is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    y = _make_dtensor(np.array([1, 1, 1], dtype=np.float32))
    z = x.add(y, alpha=3)
    assert np.allclose(z.asnumpy(), np.array([4, 5, 6]))
    assert DTensor.RUN_OPS == ["AddExt"]


# ====== Scenario 22: pow tensor-tensor (overloaded) ======

def test_pynative_fallback_pow_tensor():
    """
    Feature: Pynative fallback
    Description: Test dtensor pow (tensor exponent) fallback.
    Expectation: Result is correct and Pow is recorded.
    """
    x = _make_base_dtensor(np.array([2, 3, 4], dtype=np.float32))
    y = _make_dtensor(np.array([3, 2, 1], dtype=np.float32))
    z = x.pow(y)
    assert np.allclose(z.asnumpy(), np.array([8, 9, 4]))
    assert DTensor.RUN_OPS == ["Pow"]


# ====== Scenario 23: pow scalar (overloaded, scalar dispatch) ======

def test_pynative_fallback_pow_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor pow (scalar exponent) fallback.
    Expectation: Result is correct and PowTensorScalar is recorded.
    """
    x = _make_base_dtensor(np.array([2, 3, 4], dtype=np.float32))
    z = x.pow(2)
    assert np.allclose(z.asnumpy(), np.array([4, 9, 16]))
    assert DTensor.RUN_OPS == ["PowTensorScalar"]


# ====== Scenario 24: eq tensor-tensor (overloaded) ======

def test_pynative_fallback_eq_tensor():
    """
    Feature: Pynative fallback
    Description: Test dtensor eq (tensor-tensor) fallback.
    Expectation: Result is correct and Equal is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    y = _make_dtensor(np.array([1, 5, 3], dtype=np.float32))
    z = x.eq(y)
    assert np.array_equal(z.asnumpy(), np.array([True, False, True]))
    assert DTensor.RUN_OPS == ["Equal"]


# ====== Scenario 25: eq scalar (overloaded, scalar dispatch) ======

def test_pynative_fallback_eq_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor eq (scalar) fallback.
    Expectation: Result is correct and EqScalar is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    z = x.eq(2.0)
    assert np.array_equal(z.asnumpy(), np.array([False, True, False]))
    assert DTensor.RUN_OPS == ["EqScalar"]


# ====== Scenario 26: matmul (overloaded) ======

def test_pynative_fallback_matmul():
    """
    Feature: Pynative fallback
    Description: Test dtensor matmul fallback.
    Expectation: Result is correct and MatMulExt is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    y = _make_dtensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
    z = x.matmul(y)
    assert np.allclose(z.asnumpy(), np.array([[19, 22], [43, 50]]))
    assert DTensor.RUN_OPS == ["MatMulExt"]


# ====== Scenario 27: clamp with scalar bounds (overloaded, scalar dispatch) ======

def test_pynative_fallback_clamp_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor clamp with scalar min/max.
    Expectation: Result is clamped and ClampScalar is recorded.
    """
    x = _make_base_dtensor(np.array([1, 5, 10], dtype=np.float32))
    z = x.clamp(min=2, max=8)
    assert np.allclose(z.asnumpy(), np.array([2, 5, 8]))
    assert DTensor.RUN_OPS == ["ClampScalar"]


# ====== Scenario 28: clamp with tensor bounds (overloaded, tensor dispatch) ======

def test_pynative_fallback_clamp_tensor():
    """
    Feature: Pynative fallback
    Description: Test dtensor clamp with tensor min/max.
    Expectation: Result is clamped and ClampTensor is recorded.
    """
    x = _make_base_dtensor(np.array([1, 5, 10], dtype=np.float32))
    low = _make_dtensor(np.array([2, 2, 2], dtype=np.float32))
    high = _make_dtensor(np.array([8, 8, 8], dtype=np.float32))
    z = x.clamp(low, high)
    assert np.allclose(z.asnumpy(), np.array([2, 5, 8]))
    assert DTensor.RUN_OPS == ["ClampTensor"]


# ====== Scenario 29: neg (non-overloaded unary, pyboost_mint_class_def.tpl) ======

def test_pynative_fallback_neg():
    """
    Feature: Pynative fallback
    Description: Test dtensor neg fallback.
    Expectation: Result is correct and Neg is recorded.
    """
    x = _make_base_dtensor(np.array([1, -2, 3], dtype=np.float32))
    z = x.neg()
    assert np.allclose(z.asnumpy(), np.array([-1, 2, -3]))
    assert DTensor.RUN_OPS == ["Neg"]


# ====== Scenario 30: abs (non-overloaded unary) ======

def test_pynative_fallback_abs():
    """
    Feature: Pynative fallback
    Description: Test dtensor abs fallback.
    Expectation: Result is correct and Abs is recorded.
    """
    x = _make_base_dtensor(np.array([-1, -2, 3], dtype=np.float32))
    z = x.abs()
    assert np.allclose(z.asnumpy(), np.array([1, 2, 3]))
    assert DTensor.RUN_OPS == ["Abs"]


# ====== Scenario 31: exp (non-overloaded unary) ======

def test_pynative_fallback_exp():
    """
    Feature: Pynative fallback
    Description: Test dtensor exp fallback.
    Expectation: Result is correct and Exp is recorded.
    """
    x = _make_base_dtensor(np.array([0, 1, 2], dtype=np.float32))
    z = x.exp()
    assert np.allclose(z.asnumpy(), np.exp(np.array([0, 1, 2], dtype=np.float32)), rtol=1e-5)
    assert DTensor.RUN_OPS == ["Exp"]


# ====== Scenario 32: cos (non-overloaded unary) ======

def test_pynative_fallback_cos():
    """
    Feature: Pynative fallback
    Description: Test dtensor cos fallback.
    Expectation: Result is correct and Cos is recorded.
    """
    x = _make_base_dtensor(np.array([0, 1, 2], dtype=np.float32))
    z = x.cos()
    assert np.allclose(z.asnumpy(), np.cos(np.array([0, 1, 2], dtype=np.float32)), atol=1e-6)
    assert DTensor.RUN_OPS == ["Cos"]


# ====== Scenario 33: sqrt (non-overloaded unary) ======

def test_pynative_fallback_sqrt():
    """
    Feature: Pynative fallback
    Description: Test dtensor sqrt fallback.
    Expectation: Result is correct and Sqrt is recorded.
    """
    x = _make_base_dtensor(np.array([4, 9, 16], dtype=np.float32))
    z = x.sqrt()
    assert np.allclose(z.asnumpy(), np.array([2, 3, 4]))
    assert DTensor.RUN_OPS == ["Sqrt"]


# ====== Scenario 34: mint.cos (non-overloaded via mint function interface) ======

def test_pynative_fallback_mint_cos():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.cos function.
    Expectation: Result is correct and Cos is recorded.
    """
    x = _make_base_dtensor(np.array([0, 1, 2], dtype=np.float32))
    z = mint.cos(x)
    assert np.allclose(z.asnumpy(), np.cos(np.array([0, 1, 2], dtype=np.float32)), atol=1e-6)
    assert DTensor.RUN_OPS == ["Cos"]


# ====== Scenario 35: mean (reduction, overloaded) ======

def test_pynative_fallback_mean():
    """
    Feature: Pynative fallback
    Description: Test dtensor mean fallback.
    Expectation: Result is correct and MeanExt is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3, 4], dtype=np.float32))
    z = x.mean()
    assert np.allclose(z.asnumpy(), np.float32(2.5))
    assert DTensor.RUN_OPS == ["MeanExt"]


# ====== Scenario 36: mean with dim arg ======

def test_pynative_fallback_mean_dim():
    """
    Feature: Pynative fallback
    Description: Test dtensor mean with dim arg.
    Expectation: Result reduced along dim and MeanExt is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    z = x.mean(dim=0)
    assert np.allclose(z.asnumpy(), np.array([2, 3], dtype=np.float32))
    assert DTensor.RUN_OPS == ["MeanExt"]


# ====== Scenario 37: sum (reduction, overloaded) ======

def test_pynative_fallback_sum():
    """
    Feature: Pynative fallback
    Description: Test dtensor sum fallback.
    Expectation: Result is correct and SumExt is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3, 4], dtype=np.float32))
    z = x.sum()
    assert np.allclose(z.asnumpy(), np.float32(10))
    assert DTensor.RUN_OPS == ["SumExt"]


# ====== Scenario 38: sum with dim arg ======

def test_pynative_fallback_sum_dim():
    """
    Feature: Pynative fallback
    Description: Test dtensor sum with dim arg.
    Expectation: Result reduced along dim and SumExt is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    z = x.sum(dim=1)
    assert np.allclose(z.asnumpy(), np.array([3, 7], dtype=np.float32))
    assert DTensor.RUN_OPS == ["SumExt"]


# ====== Scenario 39: max (reduction to scalar, overloaded) ======

def test_pynative_fallback_max():
    """
    Feature: Pynative fallback
    Description: Test dtensor max (no dim) fallback.
    Expectation: Result is correct and Max is recorded.
    """
    x = _make_base_dtensor(np.array([1, 3, 2], dtype=np.float32))
    z = x.max()
    assert np.allclose(z.asnumpy(), np.float32(3))
    assert DTensor.RUN_OPS == ["Max"]


# ====== Scenario 40: max with dim (returns values+indices tuple, overloaded) ======

def test_pynative_fallback_max_dim():
    """
    Feature: Pynative fallback
    Description: Test dtensor max with dim arg (returns values, indices).
    Expectation: Values/indices are correct and MaxDim is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 3], [4, 2]], dtype=np.float32))
    values, indices = x.max(dim=1)
    assert np.allclose(values.asnumpy(), np.array([3, 4], dtype=np.float32))
    assert np.array_equal(indices.asnumpy(), np.array([1, 0]))
    assert DTensor.RUN_OPS == ["MaxDim"]


# ====== Scenario 41: squeeze (shape op, overloaded) ======

def test_pynative_fallback_squeeze():
    """
    Feature: Pynative fallback
    Description: Test dtensor squeeze fallback.
    Expectation: Shape is squeezed and Squeeze is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 2, 3]], dtype=np.float32))
    z = x.squeeze(0)
    assert z.asnumpy().shape == (3,)
    assert np.allclose(z.asnumpy(), np.array([1, 2, 3]))
    assert DTensor.RUN_OPS == ["Squeeze"]


# ====== Scenario 42: unsqueeze (shape op, overloaded) ======

def test_pynative_fallback_unsqueeze():
    """
    Feature: Pynative fallback
    Description: Test dtensor unsqueeze fallback.
    Expectation: Dim is added and ExpandDims is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    z = x.unsqueeze(0)
    assert z.asnumpy().shape == (1, 3)
    assert np.allclose(z.asnumpy(), np.array([[1, 2, 3]]))
    assert DTensor.RUN_OPS == ["ExpandDims"]


# ====== Scenario 43: flatten (shape op, overloaded) ======

def test_pynative_fallback_flatten():
    """
    Feature: Pynative fallback
    Description: Test dtensor flatten fallback.
    Expectation: Tensor is flattened and FlattenExt is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    z = x.flatten()
    assert z.asnumpy().shape == (4,)
    assert np.allclose(z.asnumpy(), np.array([1, 2, 3, 4]))
    assert DTensor.RUN_OPS == ["FlattenExt"]


# ====== Scenario 44: permute (shape op, overloaded, varargs) ======

def test_pynative_fallback_permute():
    """
    Feature: Pynative fallback
    Description: Test dtensor permute fallback.
    Expectation: Dims are transposed and Transpose is recorded.
    """
    x = _make_base_dtensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    z = x.permute(1, 0)
    assert z.asnumpy().shape == (3, 2)
    assert np.allclose(z.asnumpy(), np.array([[1, 4], [2, 5], [3, 6]]))
    assert DTensor.RUN_OPS == ["Transpose"]


# ====== Scenario 45: view (shape op, overloaded, varargs) ======

def test_pynative_fallback_view():
    """
    Feature: Pynative fallback
    Description: Test dtensor view fallback.
    Expectation: Shape is changed and Reshape is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    z = x.view(2, 3)
    assert z.asnumpy().shape == (2, 3)
    assert np.allclose(z.asnumpy(), np.array([[1, 2, 3], [4, 5, 6]]))
    assert DTensor.RUN_OPS == ["Reshape"]


# ====== Scenario 46: mint.stack with tuple (list/tuple input) ======

def test_pynative_fallback_mint_stack():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.stack (tuple input).
    Expectation: Result is stacked and StackExt is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    y = _make_dtensor(np.array([4, 5, 6], dtype=np.float32))
    z = mint.stack((x, y))
    assert np.allclose(z.asnumpy(), np.array([[1, 2, 3], [4, 5, 6]]))
    assert DTensor.RUN_OPS == ["StackExt"]


# ====== Scenario 47: mint.stack with list (list/tuple input) ======

def test_pynative_fallback_mint_stack_list():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.stack (list input).
    Expectation: Result is stacked and StackExt is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    y = _make_dtensor(np.array([4, 5, 6], dtype=np.float32))
    z = mint.stack([x, y])
    assert np.allclose(z.asnumpy(), np.array([[1, 2, 3], [4, 5, 6]]))
    assert DTensor.RUN_OPS == ["StackExt"]


# ====== Scenario 48: fill_ with scalar (in-place, overloaded) ======

def test_pynative_fallback_fill_scalar():
    """
    Feature: Pynative fallback
    Description: Test dtensor fill_ with scalar value.
    Expectation: Tensor is filled in-place and InplaceFillScalar is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    x.fill_(5.0)
    assert np.allclose(x.asnumpy(), np.array([5, 5, 5]))
    assert DTensor.RUN_OPS == ["InplaceFillScalar"]


# ====== Scenario 49: fill_ with tensor (in-place, overloaded) ======

def test_pynative_fallback_fill_tensor():
    """
    Feature: Pynative fallback
    Description: Test dtensor fill_ with tensor value.
    Expectation: Tensor is filled in-place and InplaceFillTensor is recorded.
    """
    x = _make_base_dtensor(np.array([1, 2, 3], dtype=np.float32))
    val = Tensor(np.float32(7.0))
    x.fill_(val)
    assert np.allclose(x.asnumpy(), np.array([7, 7, 7]))
    assert DTensor.RUN_OPS == ["InplaceFillTensor"]


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.01)
def test_pynative_fallback():
    """
    Feature: Pynative fallback
    Description: Run all small pynative fallback tests.
    Expectation: All sub tests run successfully.
    """
    test_pynative_fallback_add()
    test_pynative_fallback_reshape_tuple()
    test_pynative_fallback_add_scalar_overload()
    test_pynative_fallback_mint_sin()
    test_pynative_fallback_assign_primitive()
    test_pynative_fallback_mint_cat()
    test_pynative_fallback_mint_cat_list()
    test_pynative_fallback_split_primitive_attr()
    test_pynative_fallback_cast()
    test_pynative_fallback_to_dtype()
    test__update_data_shallow_copy_tensor()
    test__update_data_raise_on_non_tensor()
    # Overloaded arithmetic: tensor-tensor (pyboost_overload_mint_class_def.tpl)
    test_pynative_fallback_sub()
    test_pynative_fallback_sub_scalar()
    test_pynative_fallback_sub_alpha()
    test_pynative_fallback_operator_sub()
    test_pynative_fallback_mul()
    test_pynative_fallback_mul_scalar()
    test_pynative_fallback_operator_mul()
    test_pynative_fallback_div()
    test_pynative_fallback_div_scalar()
    test_pynative_fallback_operator_truediv()
    test_pynative_fallback_div_rounding_mode()
    test_pynative_fallback_add_alpha()
    test_pynative_fallback_pow_tensor()
    test_pynative_fallback_pow_scalar()
    test_pynative_fallback_eq_tensor()
    test_pynative_fallback_eq_scalar()
    test_pynative_fallback_matmul()
    test_pynative_fallback_clamp_scalar()
    test_pynative_fallback_clamp_tensor()
    # Non-overloaded unary ops (pyboost_mint_class_def.tpl)
    test_pynative_fallback_neg()
    test_pynative_fallback_abs()
    test_pynative_fallback_exp()
    test_pynative_fallback_cos()
    test_pynative_fallback_sqrt()
    test_pynative_fallback_mint_cos()
    # Reduction ops (overloaded)
    test_pynative_fallback_mean()
    test_pynative_fallback_mean_dim()
    test_pynative_fallback_sum()
    test_pynative_fallback_sum_dim()
    test_pynative_fallback_max()
    test_pynative_fallback_max_dim()
    # Shape ops (overloaded)
    test_pynative_fallback_squeeze()
    test_pynative_fallback_unsqueeze()
    test_pynative_fallback_flatten()
    test_pynative_fallback_permute()
    test_pynative_fallback_view()
    # List/tuple input ops
    test_pynative_fallback_mint_stack()
    test_pynative_fallback_mint_stack_list()
    # In-place ops (overloaded)
    test_pynative_fallback_fill_scalar()
    test_pynative_fallback_fill_tensor()
