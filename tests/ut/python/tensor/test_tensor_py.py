# Copyright 2020 Huawei Technologies Co., Ltd
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
"""test tensor py"""
import numpy as np
import math

import mindspore as ms
import mindspore.common.initializer as init
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine


def _attribute(tensor, shape_, size_, dtype_):
    result = (tensor.shape == shape_) and \
             (tensor.size == size_) and \
             (tensor.dtype == dtype_)
    return result


def test_tensor_init():
    nparray = np.ones([2, 2], np.float32)
    ms.Tensor(nparray)

    ms.Tensor(nparray, dtype=ms.float32)


@non_graph_engine
def test_tensor_add():
    a = ms.Tensor(np.ones([3, 3], np.float32))
    b = ms.Tensor(np.ones([3, 3], np.float32))
    a += b


@non_graph_engine
def test_tensor_sub():
    a = ms.Tensor(np.ones([2, 3]))
    b = ms.Tensor(np.ones([2, 3]))
    b -= a


@non_graph_engine
def test_tensor_mul():
    a = ms.Tensor(np.ones([3, 3]))
    b = ms.Tensor(np.ones([3, 3]))
    a *= b


def test_tensor_dim():
    arr = np.ones((1, 6))
    b = ms.Tensor(arr)
    assert b.ndim == 2


def test_tensor_size():
    arr = np.ones((1, 6))
    b = ms.Tensor(arr)
    assert arr.size == b.size


def test_tensor_itemsize():
    arr = np.ones((1, 2, 3))
    b = ms.Tensor(arr)
    assert arr.itemsize == b.itemsize


def test_tensor_strides():
    arr = np.ones((3, 4, 5, 6))
    b = ms.Tensor(arr)
    assert arr.strides == b.strides


def test_tensor_nbytes():
    arr = np.ones((3, 4, 5, 6))
    b = ms.Tensor(arr)
    assert arr.nbytes == b.nbytes


def test_dtype():
    a = ms.Tensor(np.ones((2, 3), dtype=np.int32))
    assert a.dtype == ms.int32


def test_asnumpy():
    npd = np.ones((2, 3), np.float32)
    a = ms.Tensor(npd)
    a.set_dtype(ms.float16)
    assert a.asnumpy().all() == npd.all()


def test_initializer_asnumpy():
    npd = np.ones((2, 3))
    a = init.initializer('one', [2, 3], ms.int32)
    assert a.asnumpy().all() == npd.all()


def test_print():
    a = ms.Tensor(np.ones((2, 3), np.float32))
    a.set_dtype(ms.float16)
    print(a)


def test_float():
    a = ms.Tensor(np.ones((2, 3)), ms.float16)
    assert a.dtype == ms.float16


def test_bfloat():
    """
    Feature: Test create a tensor with type of bfloat16.
    Description: Check shape/type/value of tensor with type of bfloat16.
    Expectation: success.
    """
    a = ms.Tensor(np.ones((2, 3)), ms.bfloat16)
    assert a.shape == (2, 3)
    assert a.dtype == ms.bfloat16
    a = ms.Tensor(True, ms.bfloat16)
    assert a.dtype == ms.bfloat16
    assert np.allclose(float(a.asnumpy()), 1.0)


def test_float8():
    """
    Feature: Test create a tensor with type of float8.
    Description: Check shape/type of tensor with type of float8.
    Expectation: success.
    """
    a = ms.Tensor(np.ones((2, 3)), ms.float8_e5m2)
    assert a.shape == (2, 3)
    assert a.dtype == ms.float8_e5m2
    a = ms.Tensor(np.ones((2, 3)), ms.float8_e4m3fn)
    assert a.shape == (2, 3)
    assert a.dtype == ms.float8_e4m3fn
    a = ms.Tensor(np.ones((2, 3)), ms.hifloat8)
    assert a.shape == (2, 3)
    assert a.dtype == ms.hifloat8


def test_float8_value():
    """
    Feature: Test create a tensor with type of float8.
    Description: Check shape/type/value of tensor with type of float8.
    Expectation: success.
    """
    def get_tensor_value(tensor):
        repr_str = repr(tensor)
        value_str = repr_str.split('value=')[1].strip().split(")")[0].strip()
        return float(value_str)

    max_val = ms.Tensor(57344, ms.float8_e5m2)
    assert math.isclose(get_tensor_value(max_val), 57344)
    inf_val = ms.Tensor(65535, ms.float8_e5m2)
    assert math.isinf(get_tensor_value(inf_val))
    nan_val = ms.Tensor(math.nan, ms.float8_e5m2)
    assert math.isnan(get_tensor_value(nan_val))
    min_val = ms.Tensor(math.pow(2, -16), ms.float8_e5m2)
    assert not get_tensor_value(min_val).is_integer()
    zero_val = ms.Tensor(math.pow(2, -17), ms.float8_e5m2)
    assert get_tensor_value(zero_val).is_integer()
    fp8_e5m2_val = ms.Tensor(math.pow(2, -14), ms.float8_e5m2)              # 0 00001 00
    assert math.isclose(get_tensor_value(fp8_e5m2_val), math.pow(2, -14), abs_tol=1e-10)
    fp8_e5m2_val = ms.Tensor(-math.pow(2, -14), ms.float8_e5m2)             # 1 00001 00
    assert math.isclose(get_tensor_value(fp8_e5m2_val), -math.pow(2, -14), abs_tol=1e-10)
    fp8_e5m2_val = ms.Tensor(math.pow(2, -14) * 0.75, ms.float8_e5m2)       # 0 00000 11
    assert math.isclose(get_tensor_value(fp8_e5m2_val), math.pow(2, -14) * 0.75, abs_tol=1e-10)
    fp8_e5m2_val = ms.Tensor(-math.pow(2, -14) * 0.75, ms.float8_e5m2)      # 1 00000 11
    assert math.isclose(get_tensor_value(fp8_e5m2_val), -math.pow(2, -14) * 0.75, abs_tol=1e-10)

    max_val = ms.Tensor(448, ms.float8_e4m3fn)
    assert math.isclose(get_tensor_value(max_val), 448)
    inf_val = ms.Tensor(512, ms.float8_e4m3fn)
    assert math.isnan(get_tensor_value(inf_val))
    nan_val = ms.Tensor(math.nan, ms.float8_e4m3fn)
    assert math.isnan(get_tensor_value(nan_val))
    min_val = ms.Tensor(math.pow(2, -9), ms.float8_e4m3fn)
    assert not get_tensor_value(min_val).is_integer()
    zero_val = ms.Tensor(math.pow(2, -10), ms.float8_e4m3fn)
    assert get_tensor_value(zero_val).is_integer()
    fp8_e4m3fn_val = ms.Tensor(math.pow(2, -6), ms.float8_e4m3fn)           # 0 0001 000
    assert math.isclose(get_tensor_value(fp8_e4m3fn_val), math.pow(2, -6))
    fp8_e4m3fn_val = ms.Tensor(-math.pow(2, -6), ms.float8_e4m3fn)          # 1 0001 000
    assert math.isclose(get_tensor_value(fp8_e4m3fn_val), -math.pow(2, -6))
    fp8_e4m3fn_val = ms.Tensor(math.pow(2, -6) * 0.75, ms.float8_e4m3fn)    # 0 0000 111
    assert math.isclose(get_tensor_value(fp8_e4m3fn_val), math.pow(2, -6) * 0.75, abs_tol=1e-6)
    fp8_e4m3fn_val = ms.Tensor(-math.pow(2, -6) * 0.75, ms.float8_e4m3fn)   # 1 0000 111
    assert math.isclose(get_tensor_value(fp8_e4m3fn_val), -math.pow(2, -6) * 0.75, abs_tol=1e-6)

    max_val = ms.Tensor(32768, ms.hifloat8)
    assert math.isclose(get_tensor_value(max_val), 32768)
    inf_val = ms.Tensor(65535, ms.hifloat8)
    assert math.isinf(get_tensor_value(inf_val))
    nan_val = ms.Tensor(math.nan, ms.hifloat8)
    assert math.isnan(get_tensor_value(nan_val))
    min_val = ms.Tensor(math.pow(2, -22), ms.hifloat8)
    assert not get_tensor_value(min_val).is_integer()
    zero_val = ms.Tensor(math.pow(2, -23), ms.hifloat8)
    assert get_tensor_value(zero_val).is_integer()
    hif8_val = ms.Tensor(1.125, ms.hifloat8)                    # 0 0001 001
    assert math.isclose(get_tensor_value(hif8_val), 1.125)
    hif8_val = ms.Tensor(-1.625, ms.hifloat8)                   # 1 0001 101
    assert math.isclose(get_tensor_value(hif8_val), -1.625)
    hif8_val = ms.Tensor(1.875, ms.hifloat8)                    # 0 0001 111
    assert math.isclose(get_tensor_value(hif8_val), 1.875)
    hif8_val = ms.Tensor(2, ms.hifloat8)                        # 0 001 0 000
    assert math.isclose(get_tensor_value(hif8_val), 2)
    hif8_val = ms.Tensor(-0.25, ms.hifloat8)                    # 1 001 0 001
    assert math.isclose(get_tensor_value(hif8_val), -0.25)
    hif8_val = ms.Tensor(0.5, ms.hifloat8)                      # 0 001 1 000
    assert math.isclose(get_tensor_value(hif8_val), 0.5)
    hif8_val = ms.Tensor(3.75, ms.hifloat8)                     # 0 001 0 111
    assert math.isclose(get_tensor_value(hif8_val), 3.75)
    hif8_val = ms.Tensor(-3.5, ms.hifloat8)                     # 1 001 0 110
    assert math.isclose(get_tensor_value(hif8_val), -3.5)
    hif8_val = ms.Tensor(0.9375, ms.hifloat8)                   # 0 001 1 111
    assert math.isclose(get_tensor_value(hif8_val), 0.9375)
    hif8_val = ms.Tensor(4, ms.hifloat8)                        # 0 01 00 000
    assert math.isclose(get_tensor_value(hif8_val), 4)
    hif8_val = ms.Tensor(-5, ms.hifloat8)                       # 1 01 00 010
    assert math.isclose(get_tensor_value(hif8_val), -5)
    hif8_val = ms.Tensor(0.25, ms.hifloat8)                     # 0 01 10 000
    assert math.isclose(get_tensor_value(hif8_val), 0.25)
    hif8_val = ms.Tensor(15, ms.hifloat8)                       # 0 01 01 111
    assert math.isclose(get_tensor_value(hif8_val), 15)
    hif8_val = ms.Tensor(-0.203125, ms.hifloat8)                # 1 01 11 101
    assert math.isclose(get_tensor_value(hif8_val), -0.203125)
    hif8_val = ms.Tensor(0.234375, ms.hifloat8)                 # 0 01 11 111
    assert math.isclose(get_tensor_value(hif8_val), 0.234375)
    hif8_val = ms.Tensor(16, ms.hifloat8)                       # 0 10 000 00
    assert math.isclose(get_tensor_value(hif8_val), 16)
    hif8_val = ms.Tensor(-24, ms.hifloat8)                      # 1 10 000 10
    assert math.isclose(get_tensor_value(hif8_val), -24)
    hif8_val = ms.Tensor(0.0625, ms.hifloat8)                   # 0 10 100 00
    assert math.isclose(get_tensor_value(hif8_val), 0.0625)
    hif8_val = ms.Tensor(224, ms.hifloat8)                      # 0 10 011 11
    assert math.isclose(get_tensor_value(hif8_val), 224)
    hif8_val = ms.Tensor(-math.pow(2, -7) * 1.5, ms.hifloat8)   # 1 10 111 10
    assert math.isclose(get_tensor_value(hif8_val), -math.pow(2, -7) * 1.5, abs_tol=1e-7)
    hif8_val = ms.Tensor(math.pow(2, -7), ms.hifloat8)          # 0 10 111 00
    assert math.isclose(get_tensor_value(hif8_val), math.pow(2, -7))
    hif8_val = ms.Tensor(384, ms.hifloat8)                      # 0 11 0000 1
    assert math.isclose(get_tensor_value(hif8_val), 384)
    hif8_val = ms.Tensor(-256, ms.hifloat8)                     # 1 11 0000 0
    assert math.isclose(get_tensor_value(hif8_val), -256)
    hif8_val = ms.Tensor(math.pow(2, -15), ms.hifloat8)         # 0 11 1111 0
    assert math.isclose(get_tensor_value(hif8_val), math.pow(2, -15), abs_tol=1e-10)


def test_tensor_method_sub():
    """test_tensor_method_sub"""

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub = P.Sub()

        def construct(self, x, y):
            out = x - y
            return out.transpose()

    net = Net()

    x = ms.Tensor(np.ones([5, 3], np.float32))
    y = ms.Tensor(np.ones([8, 5, 3], np.float32))
    _cell_graph_executor.compile(net, x, y)


def test_tensor_method_mul():
    """test_tensor_method_mul"""

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub = P.Sub()

        def construct(self, x, y):
            out = x * (-y)
            return out.transpose()

    net = Net()

    x = ms.Tensor(np.ones([5, 3], np.float32))
    y = ms.Tensor(np.ones([8, 5, 3], np.float32))
    _cell_graph_executor.compile(net, x, y)


def test_tensor_method_div():
    """test_tensor_method_div"""

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub = P.Sub()

        def construct(self, x, y):
            out = x / y
            return out.transpose()

    net = Net()

    x = ms.Tensor(np.ones([5, 3], np.float32))
    y = ms.Tensor(np.ones([8, 5, 3], np.float32))
    _cell_graph_executor.compile(net, x, y)


def test_asnumpy_ownership():
    """
    Feature: Tensor asnumpy() method.
    Description: Ownership should be handled correctly in asnumpy().
    Expectation: No 'use after free', no core dump.
    """
    t = init.initializer("zero", [41100, 16], dtype=ms.float32)
    t = t.init_data()
    t = t.asnumpy()
    assert np.allclose(t, 0)

    t = ms.Tensor.from_numpy(np.zeros([41100, 16], dtype=np.float32))
    t = t.asnumpy()
    assert np.allclose(t, 0)

    t = ms.Tensor(np.zeros([41100, 16], dtype=np.float32))
    t = t.asnumpy()
    assert np.allclose(t, 0)


def test_assign_value_after_asnumpy():
    """
    Feature: Tensor asnumpy() method.
    Description: Call assign_value() after asnumpy().
    Expectation: Numpy array returned from asnumpy() work as expected.
    """
    t = ms.Tensor(np.zeros([41100, 16]), ms.float32)
    n = t.asnumpy()
    c = n.copy()
    t.assign_value(ms.Tensor(np.array([6, 6, 6, 6, 6]), ms.float32))
    assert np.allclose(n, c)
    assert np.allclose(t.asnumpy(), np.array([6, 6, 6, 6, 6], np.float32))

    t = ms.Tensor.from_numpy(np.zeros([41100, 16], np.float32))
    n = t.asnumpy()
    c = n.copy()
    t.assign_value(ms.Tensor(np.array([6, 6, 6, 6, 6]), ms.float32))
    assert np.allclose(n, c)
    assert np.allclose(t.asnumpy(), np.array([6, 6, 6, 6, 6], np.float32))

    t = init.initializer("normal", [41100, 16], dtype=ms.float32)
    t = t.init_data()
    n = t.asnumpy()
    c = n.copy()
    t.assign_value(ms.Tensor(np.array([6, 6, 6, 6, 6]), ms.float32))
    assert np.allclose(n, c)
    assert np.allclose(t.asnumpy(), np.array([6, 6, 6, 6, 6], np.float32))


def test_create_np_array_from_tensor():
    """
    Feature: Tensor __array__ method.
    Description: Create numpy array from tensor.
    Expectation: Success.
    """
    n = np.array(ms.Tensor(0))
    assert n.dtype == np.int64
    assert n.shape == ()
    assert np.allclose(n, np.array(0))

    n = np.array(ms.Tensor(0, ms.float32))
    assert n.dtype == np.float32
    assert n.shape == ()
    assert np.allclose(n, np.array(0, np.float32))

    n = np.array(ms.Tensor([1, 2, 3], ms.float32))
    assert n.dtype == np.float32
    assert n.shape == (3,)
    assert np.allclose(n, np.array([1, 2, 3], np.float32))

    n = np.array([1, 2, ms.Tensor(3)])
    assert n.dtype == np.int64
    assert n.shape == (3,)
    assert np.allclose(n, np.array([1, 2, 3], np.float32))

    n = np.array([1, 2, ms.Tensor(3, ms.float32)], dtype=np.int64)
    assert n.dtype == np.int64
    assert n.shape == (3,)
    assert np.allclose(n, np.array([1, 2, 3]))

    n = np.array([[1, 2], ms.Tensor([3, 4])])
    assert n.dtype == np.int64
    assert n.shape == (2, 2)
    assert np.allclose(n, np.array([[1, 2], [3, 4]]))

    n = np.array([[1, 2], [ms.Tensor(3), 4]])
    assert n.dtype == np.int64
    assert n.shape == (2, 2)
    assert np.allclose(n, np.array([[1, 2], [3, 4]]))


def test_tensor_contains():
    """
    Feature: Tensor __contains__ method.
    Description: Test tensor __contains__ method.
    Expectation: Success.
    """
    t = ms.Tensor([1, 2])
    assert 1 in t
    assert ms.Tensor(1) in t
    assert ms.Tensor([1]) in t
    assert 3 not in t
    assert ms.Tensor(3) not in t
    assert ms.Tensor([3]) not in t
    assert 'a' not in t
    assert [1] not in t
    assert np.array(1) not in t
