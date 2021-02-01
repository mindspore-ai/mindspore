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
""" test_staging """
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore._c_expression import MetaTensor
from mindspore.common import dtype
from mindspore.common.api import ms_function
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


@ms_function
def tensor_add_func_inner(x, y):
    """ tensor_add_func_inner """
    z = F.tensor_add(x, y)
    return z


@ms_function
def tensor_add_func(x, y):
    """ tensor_add_func """
    z = tensor_add_func_inner(x, y)
    z = F.tensor_add(z, x)
    return z


@ms_function
def scalar_add(x, y):
    """ scalar_add """
    return x + y


@ms_function
def scalar_add_if(x, y):
    """ scalar_add_if """
    if x > y:
        return x + y + 10
    return x + y + 20


@ms_function
def scalar_mul_while(x):
    """ scalar_mul_while """
    rv = x
    while rv < 100:
        rv = rv * rv
    return rv


@ms_function(input_signature=(MetaTensor(dtype.float32, (1, 1, 3, 3)),
                              MetaTensor(dtype.float32, (1, 1, 3, 3))))
def tensor_add_test(x, y):
    """ tensor_add_test """
    z = F.tensor_add(x, y)
    return z


class TensorAddMulNet(nn.Cell):
    """ TensorAddMulNet definition """

    def __init__(self):
        super(TensorAddMulNet, self).__init__()
        self.add = P.Add()

    @ms_function
    def add_stage0(self, x, y):
        z = self.add(x, y)
        z = self.add(x, z)
        return z

    @ms_function
    def add_stage1(self, x, y):
        z = self.add(x, y)
        z = self.add(x, z)
        return z

    def construct(self, x, y):
        z = self.add(x, y)  # PyNative mode
        z = self.add_stage0(x, z)  # Graph mode
        z = self.add(x, z)  # PyNative mode
        z = self.add_stage1(y, z)  # Graph mode
        return z


class TensorAddNet(nn.Cell):
    """ TensorAddNet definition """

    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = P.Add()

    @ms_function
    def compute(self, x, y):
        return self.add(x, y)

    def construct(self, x, y):
        z = self.compute(x, y)
        return z


def test_control_func():
    """ test_control_func """
    res = scalar_add(3, 4)
    assert res == 7

    res = scalar_add_if(3, 4)
    assert res == 27

    res = scalar_mul_while(2)
    assert res == 256


@non_graph_engine
def test_staging_call_func():
    """ test_staging_call_func """
    x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    output = tensor_add_func(x, y)
    assert (output.asnumpy() == (np.ones([1, 1, 3, 3]) * 3)).all()


@non_graph_engine
def test_class_method_staging():
    """ test_class_method_staging """
    x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    net = TensorAddNet()
    output = net.construct(x, y)
    assert (output.asnumpy() == (np.ones([1, 1, 3, 3]) * 2)).all()


@non_graph_engine
def test_class_method_composite_staging():
    """ test_class_method_composite_staging """
    x = Tensor(np.ones([3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3]).astype(np.float32))
    net = TensorAddMulNet()
    output = net.construct(x, y)
    assert (output.asnumpy() == (np.ones([3, 3]) * 7)).astype(np.float32).all()


@non_graph_engine
def test_input_signature():
    """ test_input_signature """
    x1 = Tensor(np.ones([1, 1, 3, 3], dtype=np.float32))
    y1 = Tensor(np.ones([1, 1, 3, 3], dtype=np.float32))
    output = tensor_add_test(x1, y1)
    assert (output.asnumpy() == (np.ones([1, 1, 3, 3]) * 2)).all()
    # test input type signature
    x2 = Tensor(np.ones([1, 1, 3, 3], dtype=np.float64))
    y2 = Tensor(np.ones([1, 1, 3, 3], dtype=np.float64))
    with pytest.raises(ValueError):
        tensor_add_test(x2, y2)
    # test input shape signature
    x3 = Tensor(np.ones([1, 1, 4, 4], dtype=np.float64))
    y3 = Tensor(np.ones([1, 1, 4, 4], dtype=np.float64))
    with pytest.raises(ValueError):
        tensor_add_test(x3, y3)


def test_scalar_cast():
    """ test_scalar_cast """
    input_x = 8.5
    input_t = ms.int64

    @ms_function
    def fn_cast(x, t):
        output = F.scalar_cast(x, t)
        return output

    expect_value = 8
    z = fn_cast(input_x, input_t)
    assert z == expect_value
