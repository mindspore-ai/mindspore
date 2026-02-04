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
