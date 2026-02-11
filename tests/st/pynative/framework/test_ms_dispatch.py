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

"""Testcases for Tensor ms dispatch"""

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint, ops
from dispatch_guard import MsDispatchMode
from mindspore._c_expression import _DisableMsDispatchMode
from tests.mark_utils import arg_mark

RUN_OPS = []

class Layout():
    pass

class DTensor(Tensor):
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
    def __ms_dispatch__(cls, func, args=(), kwargs=None):
        def wrap_dtensor(x):
            if isinstance(x, Tensor):
                return DTensor(x, Layout())
            if isinstance(x, (tuple, list)):
                return type(x)(wrap_dtensor(item) for item in x)
            return x

        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}

        new_args = [arg._local_tensor if isinstance(arg, DTensor) else arg for arg in args]
        with _DisableMsDispatchMode():
            return wrap_dtensor(func(*new_args, **kwargs))


# ====== Common helper: create a DTensor from input data ======

def _make_base_dtensor(data):
    """
    Reset DTensor.RUN_OPS and create a DTensor from given data.
    """
    global RUN_OPS
    RUN_OPS = []
    x = DTensor(Tensor(data), Layout())
    return x


# ====== Scenario 1: basic add ======

def test_pynative_fallback_add():
    """
    Feature: Pynative fallback
    Description: Test dtensor add fallback.
    Expectation: Result is correct and AddExt is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    assert np.allclose(y.asnumpy(), np.array([2, 2, 2]))
    assert RUN_OPS == ["AddExt"]


# ====== Scenario 2: reshape with tuple shape ======

def test_pynative_fallback_reshape_tuple():
    """
    Feature: Pynative fallback
    Description: Test dtensor reshape with tuple shape.
    Expectation: Result is correct and Reshape is recorded after AddExt.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    assert np.allclose(z.asnumpy(), np.array([[2], [2], [2]]))
    assert RUN_OPS == ["AddExt", "Reshape"]


# ====== Scenario 3: tensor overload operator (+ scalar) ======

def test_pynative_fallback_add_scalar_overload():
    """
    Feature: Pynative fallback
    Description: Test dtensor add scalar via tensor overload.
    Expectation: Result is correct and AddScalar is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    w = z + 1
    assert isinstance(w, DTensor)
    assert np.allclose(w.asnumpy(), np.array([[3], [3], [3]]))
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar"]


# ====== Scenario 4: mint.sin ======

def test_pynative_fallback_mint_sin():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.sin.
    Expectation: Result is correct and Sin is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    w = z + 1   
    assert isinstance(w, DTensor)
    a = mint.sin(w)
    assert isinstance(a, DTensor)
    assert np.allclose(a.asnumpy(), np.sin(np.array([[3], [3], [3]])))
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin"]


# ====== Scenario 5: ops.Assign primitive ======

def test_pynative_fallback_assign_primitive():
    """
    Feature: Pynative fallback
    Description: Test dtensor with ops.Assign primitive.
    Expectation: Result is correct and Assign is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    w = z + 1
    a = mint.sin(w)
    assert isinstance(a, DTensor)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    assert isinstance(b, DTensor)
    assert np.allclose(b.asnumpy(), expect_data)
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign"]


# ====== Scenario 6: mint.cat ======

def test_pynative_fallback_mint_cat():
    """
    Feature: Pynative fallback
    Description: Test dtensor with mint.cat.
    Expectation: Result is correct and Concat is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    y = x.add(x)
    assert isinstance(y, DTensor)
    z = y.reshape((3, 1))
    assert isinstance(z, DTensor)
    w = z + 1
    assert isinstance(w, DTensor)
    a = mint.sin(w)

    expect_data = np.array([[1], [1], [1]])
    b = ops.Assign()(a, Tensor(expect_data))
    assert isinstance(b, DTensor)
    c = mint.cat((b, b))
    assert isinstance(c, DTensor)
    assert np.allclose(c.asnumpy(), np.array([[1], [1], [1], [1], [1], [1]]))
    assert RUN_OPS == ["AddExt", "Reshape", "AddScalar", "Sin", "Assign", "Concat"]


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
    assert isinstance(y[0], DTensor)
    assert isinstance(y[1], DTensor)
    assert np.allclose(y[0].asnumpy(), np.array([1, 1]))
    assert np.allclose(y[1].asnumpy(), np.array([1, 1]))
    assert RUN_OPS == ["Split"]


# ====== Scenario 8: ops.cast ======

def test_pynative_fallback_cast():
    """
    Feature: Pynative fallback
    Description: Test dtensor with ops.cast.
    Expectation: Result is correct and Cast is recorded.
    """
    x = _make_base_dtensor([1, 1, 1, 1])
    y = ms.ops.Split(axis=0, output_num=2)(x)
    assert len(y) == 2
    assert isinstance(y[0], DTensor)
    assert isinstance(y[1], DTensor)
    y0 = ms.ops.cast(y[0], ms.float32)
    assert isinstance(y0, DTensor)

    assert np.allclose(y0.asnumpy(), np.array([1.0, 1.0]))
    assert RUN_OPS == ["Split", "Cast"]


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
    assert isinstance(y1, DTensor)
    assert np.allclose(y1.asnumpy(), np.array([1.0, 1.0]))
    # to(dtype) calls Cast internally
    assert RUN_OPS == ["Split", "Cast"]


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


def test_pynative_fallback_grad():
    """
    Feature: Pynative fallback
    Description: Test grad.
    Expectation: Result is correct and Grad is recorded.
    """
    x = _make_base_dtensor([1, 1, 1])
    
    def forward_fn(x):
        return x.sin()

    ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Sin", "Cos", "Mul"]


class TestMsDispatchMode(MsDispatchMode):
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_pynative_fallback_ms_dispatch_guard():
    """
    Feature: Pynative fallback
    Description: Test MsDispatchMode.
    Expectation: Result is correct and MsDispatchMode is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestMsDispatchMode():
        x = Tensor([1, 1, 1])
        _ = x.sin()
    assert RUN_OPS == ["Sin"]


class TestNestedMsDispatchMode:
    class MsDispatchMode1(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append("MsDispatchMode1")
            RUN_OPS.append(func.name)
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)


    class MsDispatchMode2(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append("MsDispatchMode2")
            RUN_OPS.append(func.name)
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)


class ErrorRaisingGuard(MsDispatchMode):
    """Mode that raises RuntimeError on a specific op, passes through otherwise."""
    def __init__(self, error_on_op=None):
        self.error_on_op = error_on_op

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        if self.error_on_op and func.name == self.error_on_op:
            raise RuntimeError(f"Intentional error on {func.name}")
        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def test_pynative_fallback_nested_ms_dispatch_guard():
    """
    Feature: Pynative fallback
    Description: Test NestedMsDispatchMode.
    Expectation: Result is correct and NestedMsDispatchMode is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1, 1, 1])
            _ = x.sin()
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin"]


def test_pynative_fallback_nested_ms_dispatch_guard_with_dtensor():
    """
    Feature: Pynative fallback
    Description: Test NestedMsDispatchMode with DTensor.
    Expectation: Result is correct and NestedMsDispatchMode is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1, 1, 1])
            y = x.sin()
            assert isinstance(y, DTensor)
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin", "Sin"]


def test_pynative_ms_dispatch_guard_with_grad():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad.
    Expectation: Result is correct and Grad is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1, 1, 1])
    with TestMsDispatchMode():
        def forward_fn(x):
            out = x.sin()
            return out
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Sin", "Ones", "Cos", "Mul"]


def test_pynative_nested_ms_dispatch_guard_with_grad():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad.
    Expectation: Result is correct and Grad is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1, 1, 1])
            def forward_fn(x):
                return x.sin()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin", 
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones", 
                       "MsDispatchMode2", "Cos", "MsDispatchMode1", "Cos", 
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad.
    Expectation: Result is correct and Grad is recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1, 1, 1])
            def forward_fn(x):
                out = x.sin()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Sin", "MsDispatchMode1", "Sin", "Sin",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones", 
                       "MsDispatchMode2", "Cos", "MsDispatchMode1", "Cos", "Cos",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul"]


# ====== MsDispatchMode with grad: add operation ======

def test_pynative_ms_dispatch_guard_with_grad_add():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad for add operation.
    Expectation: Result is correct and AddExt grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0])
    with TestMsDispatchMode():
        def forward_fn(x):
            return x.add(x)
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["AddExt", "Ones", "Add"]


def test_pynative_nested_ms_dispatch_guard_with_grad_add():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad for add operation.
    Expectation: Result is correct and AddExt grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                return x.add(x)
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "AddExt", "MsDispatchMode1", "AddExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_add():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad for add operation.
    Expectation: Result is correct and AddExt grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                out = x.add(x)
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "AddExt", "MsDispatchMode1", "AddExt", "AddExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


# ====== MsDispatchMode with grad: reshape operation ======

def test_pynative_ms_dispatch_guard_with_grad_reshape():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad for reshape operation.
    Expectation: Result is correct and Reshape grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with TestMsDispatchMode():
        def forward_fn(x):
            return x.reshape((4,)).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Reshape", "SumExt", "Ones", "Reshape", "Tile", "Reshape"]
    assert "Reshape" in RUN_OPS


def test_pynative_nested_ms_dispatch_guard_with_grad_reshape():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad for reshape operation.
    Expectation: Result is correct and Reshape grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                return x.reshape((4,)).sum()
            ms.value_and_grad(forward_fn)(x)
    
    assert RUN_OPS == ["MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2","Reshape", "MsDispatchMode1", "Reshape"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_reshape():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad for reshape operation.
    Expectation: Result is correct and Reshape grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                out = x.reshape((4,)).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape", "Reshape",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2","Reshape", "MsDispatchMode1", "Reshape"]


# ====== MsDispatchMode with grad: mint.cat operation ======

def test_pynative_ms_dispatch_guard_with_grad_cat():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad for mint.cat operation.
    Expectation: Result is correct and Concat grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with TestMsDispatchMode():
        def forward_fn(x):
            return mint.cat((x, x), dim=0).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Concat", "SumExt", "Ones", "Reshape", "Tile", "SplitWithSize", "Add"]


def test_pynative_nested_ms_dispatch_guard_with_grad_cat():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad for mint.cat operation.
    Expectation: Result is correct and Concat grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                return mint.cat((x, x), dim=0).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Concat", "MsDispatchMode1", "Concat",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "SplitWithSize", "MsDispatchMode1", "SplitWithSize",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_cat():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad for mint.cat operation.
    Expectation: Result is correct and Concat grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([[1.0, 2.0], [3.0, 4.0]])
            def forward_fn(x):
                out = mint.cat((x, x), dim=0).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Concat", "MsDispatchMode1", "Concat", "Concat",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "SplitWithSize", "MsDispatchMode1", "SplitWithSize",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


# ====== MsDispatchMode with grad: ops.cast operation ======

def test_pynative_ms_dispatch_guard_with_grad_cast():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad for ops.cast operation.
    Expectation: Result is correct and Cast grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0], dtype=ms.float32)
    with TestMsDispatchMode():
        def forward_fn(x):
            y = ms.ops.cast(x, ms.float16)
            return ms.ops.cast(y, ms.float32).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Cast", "Cast", "SumExt", "Ones", "Reshape", "Tile", "Cast", "Cast"]


def test_pynative_nested_ms_dispatch_guard_with_grad_cast():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad for ops.cast operation.
    Expectation: Result is correct and Cast grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0], dtype=ms.float32)
            def forward_fn(x):
                y = ms.ops.cast(x, ms.float16)
                return ms.ops.cast(y, ms.float32).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_cast():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad for ops.cast operation.
    Expectation: Result is correct and Cast grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            def forward_fn(x):
                y = ms.ops.cast(x, ms.float16)
                out = ms.ops.cast(y, ms.float32).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast", "Cast",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast",
                       "MsDispatchMode2", "Cast", "MsDispatchMode1", "Cast"]


# ====== MsDispatchMode with grad: mul operation ======

def test_pynative_ms_dispatch_guard_with_grad_mul():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad for mul operation.
    Expectation: Result is correct and Mul grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0])
    with TestMsDispatchMode():
        def forward_fn(x):
            return (x * x).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Mul", "SumExt", "Ones", "Reshape", "Tile", "Mul", "Mul", "Add"]


def test_pynative_nested_ms_dispatch_guard_with_grad_mul():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad for mul operation.
    Expectation: Result is correct and Mul grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                return (x * x).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_mul():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad for mul operation.
    Expectation: Result is correct and Mul grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                out = (x * x).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul",
                       "MsDispatchMode2", "Add", "MsDispatchMode1", "Add", "Add"]


# ====== MsDispatchMode with grad: exp operation ======

def test_pynative_ms_dispatch_guard_with_grad_exp():
    """
    Feature: Pynative fallback
    Description: Test ms dispatch guard with grad for exp operation.
    Expectation: Result is correct and Exp grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = Tensor([1.0, 2.0, 3.0])
    with TestMsDispatchMode():
        def forward_fn(x):
            return mint.exp(x).sum()
        ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["Exp", "SumExt", "Ones", "Reshape", "Tile", "Mul"]


def test_pynative_nested_ms_dispatch_guard_with_grad_exp():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard with grad for exp operation.
    Expectation: Result is correct and Exp grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = Tensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                return mint.exp(x).sum()
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Exp", "MsDispatchMode1", "Exp",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul"]


def test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_exp():
    """
    Feature: Pynative fallback
    Description: Test nested ms dispatch guard and dtensor with grad for exp operation.
    Expectation: Result is correct and Exp grad ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestNestedMsDispatchMode.MsDispatchMode1():
        with TestNestedMsDispatchMode.MsDispatchMode2():
            x = _make_base_dtensor([1.0, 2.0, 3.0])
            def forward_fn(x):
                out = mint.exp(x).sum()
                assert isinstance(out, DTensor)
                return out
            ms.value_and_grad(forward_fn)(x)
    assert RUN_OPS == ["MsDispatchMode2", "Exp", "MsDispatchMode1", "Exp", "Exp",
                       "MsDispatchMode2", "SumExt", "MsDispatchMode1", "SumExt", "SumExt",
                       "MsDispatchMode2", "Ones", "MsDispatchMode1", "Ones",
                       "MsDispatchMode2", "Reshape", "MsDispatchMode1", "Reshape",
                       "MsDispatchMode2", "Tile", "MsDispatchMode1", "Tile",
                       "MsDispatchMode2", "Mul", "MsDispatchMode1", "Mul", "Mul"]


def test_pynative_disable_ms_dispatch_guard_bypasses_dispatch():
    """
    Feature: Pynative fallback
    Description: _DisableMsDispatchMode suppresses mode dispatch.
    Expectation: Ops inside _DisableMsDispatchMode are NOT recorded by the outer mode.
    """
    global RUN_OPS
    RUN_OPS = []
    with TestMsDispatchMode():
        x = Tensor([1.0, 2.0, 3.0])
        _ = x.sin()  # should be recorded
        with _DisableMsDispatchMode():
            _ = x.cos()  # should NOT be recorded
        _ = x.exp()  # should be recorded
    assert "Sin" in RUN_OPS
    assert "Cos" not in RUN_OPS
    assert "Exp" in RUN_OPS


def test_pynative_disable_ms_dispatch_guard_with_dtensor():
    """
    Feature: Pynative fallback
    Description: _DisableMsDispatchMode suppresses DTensor dispatch.
    Expectation: Ops inside _DisableMsDispatchMode bypass DTensor.__ms_dispatch__
                 and return a plain Tensor.
    """
    global RUN_OPS
    RUN_OPS = []
    x = _make_base_dtensor([1.0, 2.0, 3.0])
    y = x.sin()
    assert isinstance(y, DTensor)
    assert "Sin" in RUN_OPS
    ops_before = len(RUN_OPS)
    with _DisableMsDispatchMode():
        z = Tensor([1.0, 2.0, 3.0]).sin()
    assert not isinstance(z, DTensor)
    assert len(RUN_OPS) == ops_before  # no new ops recorded


def test_pynative_guard_exception_handling():
    """
    Feature: Pynative fallback
    Description: Exception propagation + guard stays functional after catch.
    Expectation: RuntimeError propagates; guard works normally after exception is caught.
    """
    global RUN_OPS
    RUN_OPS = []
    guard = ErrorRaisingGuard(error_on_op="Cos")
    with guard:
        x = Tensor([1.0, 2.0, 3.0])
        _ = x.sin()  # should succeed
        assert "Sin" in RUN_OPS
        try:
            _ = x.cos()  # should raise
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "Intentional error on Cos" in str(e)
        # Mode should still work after the caught exception
        RUN_OPS.clear()
        _ = x.exp()
        assert "Exp" in RUN_OPS


def test_pynative_guard_created_outside_with_block():
    """
    Feature: Pynative fallback
    Description: Mode instance created before `with` block.
    Expectation: Mode dispatches correctly when used with a pre-created instance.
    """
    global RUN_OPS
    RUN_OPS = []
    guard = TestMsDispatchMode()
    x = Tensor([1.0, 2.0, 3.0])
    # Mode not active yet
    _ = x.sin()
    assert RUN_OPS == []
    # Now activate it
    with guard:
        _ = x.cos()
    assert RUN_OPS == ["Cos"]


def test_pynative_nesting_same_guard_instance():
    """
    Feature: Pynative fallback
    Description: Same guard instance pushed twice on stack.
    Expectation: Either works with double dispatch, or raises RuntimeError.
    """
    global RUN_OPS
    RUN_OPS = []
    guard = TestMsDispatchMode()
    try:
        with guard:
            with guard:
                x = Tensor([1.0, 2.0, 3.0])
                _ = x.sin()
        # If it works, Sin should appear twice (once per guard level)
        assert RUN_OPS.count("Sin") == 2
    except RuntimeError:
        # MindSpore may reject pushing the same guard instance twice
        pass


def test_pynative_guard_priority_over_dtensor():
    """
    Feature: Pynative fallback
    Description: Mode dispatches before DTensor.__ms_dispatch__.
    Expectation: Mode's __ms_dispatch__ is called first, then DTensor's.
    """
    global RUN_OPS
    RUN_OPS = []

    class PriorityGuard(MsDispatchMode):
        def __ms_dispatch__(self, func, args=(), kwargs=None):
            RUN_OPS.append("PriorityGuard")
            RUN_OPS.append(func.name)
            if kwargs is None:
                kwargs = {}
            return func(*args, **kwargs)

    with PriorityGuard():
        x = _make_base_dtensor([1.0, 2.0, 3.0])
        _ = x.sin()
    # Mode should appear first, then DTensor's dispatch
    assert RUN_OPS[0] == "PriorityGuard"
    assert "Sin" in RUN_OPS


def test_pynative_grad_numerical_correctness():
    """
    Feature: Pynative fallback
    Description: Verify actual gradient values, not just RUN_OPS.
    Expectation: Gradient of sin(x) is cos(x); values match numerically.
    """
    global RUN_OPS
    RUN_OPS = []
    data = np.array([0.0, np.pi / 4, np.pi / 2], dtype=np.float32)
    x = DTensor(Tensor(data), Layout())

    def forward_fn(x):
        return x.sin().sum()

    value, grad = ms.value_and_grad(forward_fn)(x)
    expected_grad = np.cos(data)
    assert np.allclose(grad.asnumpy(), expected_grad, atol=1e-5)
    expected_value = np.sum(np.sin(data))
    assert np.allclose(value.asnumpy(), expected_value, atol=1e-5)


def test_pynative_guard_as_decorator():
    """
    Feature: Pynative fallback
    Description: MsDispatchMode used as @decorator (ContextDecorator).
    Expectation: Mode is active during the decorated function call.
    """
    global RUN_OPS
    RUN_OPS = []

    @TestMsDispatchMode()
    def guarded_computation():
        x = Tensor([1.0, 2.0, 3.0])
        return x.sin()

    result = guarded_computation()
    assert np.allclose(result.asnumpy(), np.sin(np.array([1.0, 2.0, 3.0])), atol=1e-5)
    assert "Sin" in RUN_OPS


def test_pynative_fallback_dtensor_mul_scalar_inplace_pattern():
    """
    Feature: Pynative fallback
    Description: DTensor with multiply-accumulate pattern.
    Expectation: Result is numerically correct and mul ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = DTensor(Tensor([1.0, 2.0, 3.0]), Layout())
    y = x * 2
    z = y + x
    assert isinstance(y, DTensor)
    assert isinstance(z, DTensor)
    assert np.allclose(z.asnumpy(), np.array([3.0, 6.0, 9.0]))
    # Check that mul and add ops were recorded
    mul_ops = [op for op in RUN_OPS if "Mul" in op]
    add_ops = [op for op in RUN_OPS if "Add" in op]
    assert len(mul_ops) >= 1
    assert len(add_ops) >= 1


def test_pynative_complex_multi_branch_computation():
    """
    Feature: Pynative fallback
    Description: Diamond-shaped computation graph (fan-out/fan-in).
    Expectation: Result is numerically correct; all branch ops are recorded.
    """
    global RUN_OPS
    RUN_OPS = []
    x = DTensor(Tensor([1.0, 2.0, 3.0]), Layout())
    # Fan-out: two branches from x
    branch_a = x.sin()
    branch_b = x.cos()
    # Fan-in: combine branches
    result = branch_a.add(branch_b)
    assert isinstance(branch_a, DTensor)
    assert isinstance(branch_b, DTensor)
    assert isinstance(result, DTensor)
    expected = np.sin(np.array([1.0, 2.0, 3.0])) + np.cos(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(result.asnumpy(), expected, atol=1e-5)
    assert "Sin" in RUN_OPS
    assert "Cos" in RUN_OPS
    assert "AddExt" in RUN_OPS


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
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
    test_pynative_fallback_split_primitive_attr()
    test_pynative_fallback_cast()
    test_pynative_fallback_to_dtype()
    test__update_data_shallow_copy_tensor()
    test__update_data_raise_on_non_tensor()

    test_pynative_fallback_ms_dispatch_guard()
    test_pynative_fallback_nested_ms_dispatch_guard()
    test_pynative_fallback_nested_ms_dispatch_guard_with_dtensor()

    test_pynative_ms_dispatch_guard_with_grad()
    test_pynative_nested_ms_dispatch_guard_with_grad()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad()

    # MsDispatchMode with grad: add operation
    test_pynative_ms_dispatch_guard_with_grad_add()
    test_pynative_nested_ms_dispatch_guard_with_grad_add()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_add()

    # MsDispatchMode with grad: reshape operation
    test_pynative_ms_dispatch_guard_with_grad_reshape()
    test_pynative_nested_ms_dispatch_guard_with_grad_reshape()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_reshape()

    # MsDispatchMode with grad: mint.cat operation
    test_pynative_ms_dispatch_guard_with_grad_cat()
    test_pynative_nested_ms_dispatch_guard_with_grad_cat()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_cat()

    # MsDispatchMode with grad: ops.cast operation
    test_pynative_ms_dispatch_guard_with_grad_cast()
    test_pynative_nested_ms_dispatch_guard_with_grad_cast()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_cast()

    # MsDispatchMode with grad: mul operation
    test_pynative_ms_dispatch_guard_with_grad_mul()
    test_pynative_nested_ms_dispatch_guard_with_grad_mul()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_mul()

    # MsDispatchMode with grad: exp operation
    test_pynative_ms_dispatch_guard_with_grad_exp()
    test_pynative_nested_ms_dispatch_guard_with_grad_exp()
    test_pynative_nested_ms_dispatch_guard_and_dtensor_with_grad_exp()

    # New scenario tests
    test_pynative_disable_ms_dispatch_guard_bypasses_dispatch()
    test_pynative_disable_ms_dispatch_guard_with_dtensor()
    test_pynative_guard_exception_handling()
    test_pynative_guard_created_outside_with_block()
    test_pynative_nesting_same_guard_instance()
    test_pynative_guard_priority_over_dtensor()
    test_pynative_grad_numerical_correctness()
    test_pynative_guard_as_decorator()
    test_pynative_fallback_dtensor_mul_scalar_inplace_pattern()
    test_pynative_complex_multi_branch_computation()
