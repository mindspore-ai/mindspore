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

import mindspore as ms
import mindspore.common.initializer as init
from mindspore.common.api import _executor
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
    npd = np.ones((2, 3))
    a = ms.Tensor(npd)
    a.set_dtype(ms.int32)
    assert a.asnumpy().all() == npd.all()


def test_initializer_asnumpy():
    npd = np.ones((2, 3))
    a = init.initializer('one', [2, 3], ms.int32)
    assert a.asnumpy().all() == npd.all()


def test_print():
    a = ms.Tensor(np.ones((2, 3)))
    a.set_dtype(ms.int32)
    print(a)


def test_float():
    a = ms.Tensor(np.ones((2, 3)), ms.float16)
    assert a.dtype == ms.float16


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
    _executor.compile(net, x, y)


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
    _executor.compile(net, x, y)


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
    _executor.compile(net, x, y)
