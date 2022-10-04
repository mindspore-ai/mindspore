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
"""
@File  : test_tensor.py
@Author:
@Date  : 2019-03-14
@Desc  : test mindspore tensor's operation
"""
import numpy as np
import pytest

import mindspore as ms
import mindspore.common.api as me
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from ..ut_filter import non_graph_engine

ndarr = np.ones((2, 3))

context.set_context(mode=context.GRAPH_MODE)


def test_tensor_flatten():
    with pytest.raises(AttributeError):
        lst = [1, 2, 3, 4,]
        tensor_list = ms.Tensor(lst, ms.float32)
        tensor_list = tensor_list.Flatten()
        print(tensor_list)


def test_tensor_list():
    lst = [[1.0, 2.0, 1.0], [1.0, 10.0, 9.0]]
    tensor_list = ms.Tensor(lst, ms.float32)
    print(tensor_list)


def test_tensor():
    """test_tensor"""
    t1 = ms.Tensor(ndarr)
    assert isinstance(t1, ms.Tensor)
    assert t1.dtype == ms.float64

    t2 = ms.Tensor(np.zeros([1, 2, 3]), ms.float32)
    assert isinstance(t2, ms.Tensor)
    assert t2.shape == (1, 2, 3)
    assert t2.dtype == ms.float32

    t3 = ms.Tensor(0.1)
    assert isinstance(t3, ms.Tensor)
    assert t3.dtype == ms.float32

    t4 = ms.Tensor(1)
    assert isinstance(t4, ms.Tensor)
    assert t4.dtype == ms.int64


def test_tensor_type_float16():
    t_float16 = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16))
    assert isinstance(t_float16, ms.Tensor)
    assert t_float16.shape == (2, 3)
    assert t_float16.dtype == ms.float16

def test_tensor_type_complex64():
    np_input = np.array(
        [[1+0.1j, 2j, 3+0.3j], [4-0.4j, 5, 6]], dtype=np.complex64)
    t_complex64 = ms.Tensor(np_input)
    assert isinstance(t_complex64, ms.Tensor)
    assert t_complex64.shape == (2, 3)
    assert t_complex64.dtype == ms.complex64
    assert np.all(t_complex64.asnumpy() == np_input)


def test_tensor_type_complex64_user_define():
    np_input = np.zeros([1, 2, 3])
    t_complex64 = ms.Tensor(np_input, ms.complex64)
    assert isinstance(t_complex64, ms.Tensor)
    assert t_complex64.shape == (1, 2, 3)
    assert t_complex64.dtype == ms.complex64
    assert np.all(t_complex64.asnumpy() == np_input)


def test_tensor_type_complex128():
    #complex python object
    py_input = 1 + 2.22222222j
    t_complex128 = ms.Tensor(py_input)
    assert t_complex128.shape == ()
    assert t_complex128.dtype == ms.complex128
    assert np.all(t_complex128.asnumpy() == py_input)

    #complex in numpy array
    np_input = np.array(
        [[1+0.1j, 2j, 3+0.3j], [4-0.4j, 5, 6]], dtype=np.complex128)
    t_complex128 = ms.Tensor(np_input)
    assert isinstance(t_complex128, ms.Tensor)
    assert t_complex128.shape == (2, 3)
    assert t_complex128.dtype == ms.complex128
    assert np.all(t_complex128.asnumpy() == np_input)

    #complex in tuple
    py_input = (1, 2.22222222j, 3)
    t_complex128 = ms.Tensor(py_input)
    assert np.all(t_complex128.asnumpy() == py_input)

    #complex in list
    py_input = [[1+0.1j, 2j, 3+0.3j], [4-0.4j, 5, 6]]
    t_complex128 = ms.Tensor(py_input)
    assert isinstance(t_complex128, ms.Tensor)
    assert t_complex128.shape == (2, 3)
    assert t_complex128.dtype == ms.complex128
    assert np.all(t_complex128.asnumpy() == py_input)

def test_tensor_type_complex128_user_define():
    np_input = np.zeros([1, 2, 3])
    t_complex128 = ms.Tensor(np_input, ms.complex128)
    assert isinstance(t_complex128, ms.Tensor)
    assert t_complex128.shape == (1, 2, 3)
    assert t_complex128.dtype == ms.complex128
    assert np.all(t_complex128.asnumpy() == np_input)

def test_tensor_type_float32():
    t_float32 = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    assert isinstance(t_float32, ms.Tensor)
    assert t_float32.shape == (2, 3)
    assert t_float32.dtype == ms.float32


def test_tensor_type_float32_user_define():
    t = ms.Tensor(np.zeros([1, 2, 3]), ms.float32)
    assert isinstance(t, ms.Tensor)
    assert t.shape == (1, 2, 3)
    assert t.dtype == ms.float32


def test_tensor_type_float64():
    t = ms.Tensor([[1.0, 2, 3], [4, 5, 6]])
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == ms.float32

    t_zero = ms.Tensor(np.zeros([1, 2, 3]))
    assert isinstance(t_zero, ms.Tensor)
    assert t_zero.shape == (1, 2, 3)
    assert t_zero.dtype == ms.float64


def test_tensor_type_float64_user_define():
    t = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == ms.float64

    t_float64 = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), ms.float64)
    assert isinstance(t_float64, ms.Tensor)
    assert t_float64.shape == (2, 3)
    assert t_float64.dtype == ms.float64


def test_tensor_type_bool():
    # init a tensor with bool type
    ts_bool_array = ms.Tensor(np.zeros([2, 3], np.bool), ms.bool_)
    assert isinstance(ts_bool_array, ms.Tensor)
    assert ts_bool_array.dtype == ms.bool_

    t_bool = ms.Tensor(True)
    assert isinstance(t_bool, ms.Tensor)
    assert t_bool.dtype == ms.bool_

    t_bool_array = ms.Tensor(np.array([[True, False, True], [False, False, False]]))
    assert isinstance(t_bool_array, ms.Tensor)
    assert t_bool_array.shape == (2, 3)
    assert t_bool_array.dtype == ms.bool_


def test_tensor_type_int8():
    t_int8_array = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8))
    assert isinstance(t_int8_array, ms.Tensor)
    assert t_int8_array.shape == (2, 3)
    assert t_int8_array.dtype == ms.int8


def test_tensor_type_int16():
    t_int16_array = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16))
    assert isinstance(t_int16_array, ms.Tensor)
    assert t_int16_array.shape == (2, 3)
    assert t_int16_array.dtype == ms.int16


def test_tensor_type_int32():
    t_int = ms.Tensor([[1, 2, 3], [4, 5, 6]])
    assert isinstance(t_int, ms.Tensor)
    assert t_int.shape == (2, 3)
    assert t_int.dtype == ms.int64

    t_int_array = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    assert isinstance(t_int_array, ms.Tensor)
    assert t_int_array.shape == (2, 3)
    assert t_int_array.dtype == ms.int32


def test_tensor_type_int64():
    t_int64 = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
    assert isinstance(t_int64, ms.Tensor)
    assert t_int64.shape == (2, 3)
    assert t_int64.dtype == ms.int64


def test_tensor_type_uint8():
    t_uint8_array = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))
    assert isinstance(t_uint8_array, ms.Tensor)
    assert t_uint8_array.shape == (2, 3)
    assert t_uint8_array.dtype == ms.uint8


def test_tensor_type_uint16():
    t_uint16_array = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16))
    assert isinstance(t_uint16_array, ms.Tensor)
    assert t_uint16_array.shape == (2, 3)
    assert t_uint16_array.dtype == ms.uint16


def test_tensor_type_uint32():
    t_uint32_array = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32))
    assert isinstance(t_uint32_array, ms.Tensor)
    assert t_uint32_array.shape == (2, 3)
    assert t_uint32_array.dtype == ms.uint32


def test_tensor_type_uint64():
    t_uint64 = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64))
    assert isinstance(t_uint64, ms.Tensor)
    assert t_uint64.shape == (2, 3)
    assert t_uint64.dtype == ms.uint64


def test_set_type():
    t = ms.Tensor(ndarr)
    t.set_dtype(ms.float32)
    assert t.dtype == ms.float32


@non_graph_engine
def test_add():
    x = ms.Tensor(ndarr)
    y = ms.Tensor(ndarr)
    z = x + y
    assert isinstance(z, ms.Tensor)


@non_graph_engine
def test_sub():
    x = ms.Tensor(ndarr)
    y = ms.Tensor(ndarr)
    z = x - y
    assert isinstance(z, ms.Tensor)


@non_graph_engine
def test_div():
    x = ms.Tensor(np.array([[2, 6, 10], [12, 4, 8]]).astype(np.float32))
    y = ms.Tensor(np.array([[2, 2, 5], [6, 1, 2]]).astype(np.float32))
    z = x / y
    z2 = x / 2
    assert isinstance(z, ms.Tensor)
    assert isinstance(z2, ms.Tensor)


@non_graph_engine
def test_parameter():
    x = Parameter(initializer(1, [1], ms.float32), name="beta1_power")
    x = x.init_data()
    z = x / 2
    print(z)


class Net(nn.Cell):
    """Net definition"""

    def __init__(self, dim):
        super(Net, self).__init__()
        self.dim = dim

    def construct(self, input_x):
        return input_x


@non_graph_engine
def test_return_tensor():
    """test_return_tensor"""
    net = Net(0)
    input_data = ms.Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    input_data.set_dtype(ms.float32)
    exe = me._cell_graph_executor
    exe.compile(net, input_data)
    tensor_ = exe(net, input_data)

    # get shape
    shape_ = tensor_.shape
    print("shape = ", shape_)

    # get type
    type_ = tensor_.dtype
    print("type = ", type_)

    # get value
    value_ = tensor_.asnumpy()
    print("numpy value = ", value_)


def test_tensor_contiguous():
    """test_tensor_contiguous"""
    input_c = np.arange(6).reshape(2, 3)
    input_f = input_c.T
    np.ascontiguousarray(input_c, dtype=np.float32)
    assert True, input_c.flags['C_CONTIGUOUS']

    print("input_f flags = ", input_f.flags)
    assert True, input_f.flags['F_CONTIGUOUS']

    tensor_f_float32 = ms.Tensor(input_f)
    rt_f = tensor_f_float32.asnumpy()
    assert True, rt_f.flags['C_CONTIGUOUS']
    print("rt_f flags = ", rt_f.flags)


def test_tensor_contiguous2():
    input_data = np.random.randn(32, 112, 112, 3).astype(np.float32)
    input_me = input_data.transpose(0, 3, 1, 2)
    print("input_me flags = ", input_me.flags)
    tensor_f_float32 = ms.Tensor(input_me)
    out_f = tensor_f_float32.asnumpy()
    print("out_f flags = ", out_f.flags)


def test_tensor_input_string():
    with pytest.raises(TypeError):
        input_data = 'ccc'
        ms.Tensor(input_data)


def test_tensor_input_tuple_string():
    with pytest.raises(TypeError):
        input_data = (2, 3, '4', 5)
        ms.Tensor(input_data)


def test_tensor_input_list_string():
    with pytest.raises(TypeError):
        input_data = [[2, 3, '4', 5], [1, 2, 3, 4]]
        ms.Tensor(input_data)


def test_tensor_input_none():
    with pytest.raises(TypeError):
        input_data = None
        ms.Tensor(input_data, np.int64)


# pylint: disable=no-value-for-parameter
def test_tensor_input_empty():
    with pytest.raises(TypeError):
        ms.Tensor()


def test_tensor_input_ndarray_str():
    inp = np.array(["88", 0, 9])
    tensor = ms.Tensor(inp)
    assert str(tensor) == "Tensor(shape=[3], dtype=String, " \
                          "value= ['88', '0', '9'])"


def test_tensor_input_ndarray_bool():
    inp = np.array([True, 2, 4])
    ms.Tensor(inp)

    inp = np.array([False, 2, 4])
    ms.Tensor(inp)

def test_tensor_input_ndarray_none():
    with pytest.raises(TypeError):
        inp = np.array([None, 2, 4])
        ms.Tensor(inp)


def test_tensor_input_ndarray_dict():
    with pytest.raises(TypeError):
        inp = {'a': 6, 'b': 7}
        ms.Tensor(inp)


def test_tensor_input_np_nan():
    with pytest.raises(TypeError):
        input_data = (1, 2, 3, np.nan)
        ms.Tensor(input_data, np.int64)


def test_tensor_input_tuple_inf():
    with pytest.raises(TypeError):
        input_data = (1, 2, 3, float("inf"))
        ms.Tensor(input_data, np.int64)


def test_tensor_input_dict():
    with pytest.raises(TypeError):
        input_data = {'a': 6, 'b': 7}
        ms.Tensor(input_data, np.int64)


def test_tensor_input_complex():
    with pytest.raises(TypeError):
        input_data = (1, 2j, 3)
        ms.Tensor(input_data, np.int64)


def test_tensor_dtype_np_float():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.float)
        ms.Tensor(input_data, np.float)


def test_tensor_dtype_np_float16():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.float16)
        ms.Tensor(input_data, np.float16)


def test_tensor_dtype_np_float32():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.float32)
        ms.Tensor(input_data, np.float32)


def test_tensor_dtype_np_float64():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.float64)
        ms.Tensor(input_data, np.float64)


def test_tensor_dtype_np_int():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.int)
        ms.Tensor(input_data, np.int)


def test_tensor_dtype_np_int8():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.int8)
        ms.Tensor(input_data, np.int8)


def test_tensor_dtype_np_int16():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.int16)
        ms.Tensor(input_data, np.int16)


def test_tensor_dtype_np_int32():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.int32)
        ms.Tensor(input_data, np.int32)


def test_tensor_dtype_np_int64():
    with pytest.raises(TypeError):
        input_data = np.random.randn(32, 112, 112, 3).astype(np.int64)
        ms.Tensor(input_data, np.int64)


def test_tensor_dtype_fp32_to_bool():
    input_ = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_ = ms.Tensor(input_)
    t = ms.Tensor(input_, dtype=ms.bool_)
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3, 4, 5)
    assert t.dtype == ms.bool_


def test_tensor_dtype_fp64_to_uint8():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    t = ms.Tensor(array, ms.uint8)
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == ms.uint8

def test_tensor_dtype_complex64_to_float32():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
    t = ms.Tensor(array, ms.float32)
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == ms.float32

def test_tensor_dtype_float32_to_complex64():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = ms.Tensor(array, ms.complex64)
    assert isinstance(t, ms.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == ms.complex64

def test_tensor_operation():
    x = Tensor(np.ones((3, 3)) * 4)
    res = x + 1
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 5)
    res = 1 + x
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 5)
    res = x - 2
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 2)
    res = 6 - x
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 2)
    res = x * 3
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 12)
    res = 3 * x
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 12)
    res = x / 2
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 2)
    res = 8 / x
    assert np.all(res.asnumpy() == np.ones((3, 3)) * 2)
    res = x % 3
    assert np.all(res.asnumpy() == np.ones((3, 3)))
    res = x // 3
    assert np.all(res.asnumpy() == np.ones((3, 3)))
    x %= 3
    assert np.all(x.asnumpy() == np.ones((3, 3)))
    res = x * (2, 3, 4)
    assert np.all(res.asnumpy() == np.ones((3, 3)) * (2, 3, 4))
    res = 5 % x
    assert np.all(x.asnumpy() == np.ones((3, 3)))
    res = 5 // x
    assert np.all(x.asnumpy() == np.ones((3, 3)))

def test_tensor_from_numpy():
    a = np.ones((2, 3))
    t = ms.Tensor.from_numpy(a)
    assert isinstance(t, ms.Tensor)
    assert np.all(t.asnumpy() == 1)
    # 't' and 'a' share same data.
    a[1] = 2
    assert np.all(t.asnumpy()[0] == 1)
    assert np.all(t.asnumpy()[1] == 2)
    # 't' is still valid after 'a' deleted.
    del a
    assert np.all(t.asnumpy()[0] == 1)
    assert np.all(t.asnumpy()[1] == 2)
    with pytest.raises(TypeError):
        # incorrect input.
        t = ms.Tensor.from_numpy([1, 2, 3])

    x = np.array([[1, 2], [3, 4]], order='F')
    b = Tensor.from_numpy(x)
    assert np.all(b.asnumpy() == np.array([[1, 2], [3, 4]]))
