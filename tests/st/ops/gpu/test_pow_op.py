# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import random
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, ops
from mindspore.nn import Cell
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Pow()

    def construct(self, x, y):
        return self.ops(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_int_neg_exp():
    """
    Feature: Support negative exponent when inputs are int datatypes.
    Description: Input signed int numbers to Pow op. Test whether it supports negative exponents.
    Expectation: Output of Pow op match to expectations. (numpy.power not support negative exponents)
    """
    int_datatypes = (np.int8, np.int16, np.int32, np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net()
    for datatype in int_datatypes:
        base = Tensor(np.array([[2, 2], [-3, 3], [-1, -1], [0, 0]], datatype))
        exp = Tensor(np.array([-1, -2], datatype))
        out = net(base, exp).asnumpy()
        expect = np.array([[0, 0], [0, 0], [-1, 1], [0, 0]])
        assert np.allclose(out, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_int_real_datatypes():
    """
    Feature: Support various int and real datatypes.
    Description: Input various types of int and real numbers to Pow op. Test whether it supports.
    Expectation: Output of Pow op match to numpy.power.
    """
    real_datatypes = (np.uint8, np.uint16, np.uint32, np.uint64,
                      np.int8, np.int16, np.int32, np.int64,
                      np.float16, np.float32, np.float64)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net()
    for datatype in real_datatypes:
        x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(datatype)
        y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(datatype)
        x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(datatype)
        y1_np = np.array(3).astype(datatype)

        x0 = Tensor(x0_np)
        y0 = Tensor(y0_np)
        x1 = Tensor(x1_np)
        y1 = Tensor(y1_np)

        out = net(x0, y0).asnumpy()
        expect = np.power(x0_np, y0_np)
        assert np.allclose(out, expect)
        assert out.shape == expect.shape

        out = net(x1, y1).asnumpy()
        expect = np.power(x1_np, y1_np)
        assert np.allclose(out, expect)
        assert out.shape == expect.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_complex_datatypes():
    """
    Feature: Support various complex datatypes.
    Description: Input various types of complex numbers input Pow op. Test whether it supports.
    Expectation: Output of Pow op match to numpy.power.
    """
    complex_datatypes = (np.complex64, np.complex128)
    real_datatypes = (ms.float32, ms.float64)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net()
    for datatype, real_datatype in zip(complex_datatypes, real_datatypes):
        x0_np = np.array([1.+1.j, 1.-1.j, -1.+1.j, -1.-1.j]).astype(datatype)
        y0_np = np.array([1.+1.j, 1.-1.j, -1.+1.j, -1.-1.j]).astype(datatype)
        x1_np = np.array([1.+1.j, 1.-1.j, -1.+1.j, -1.-1.j]).astype(datatype)
        y1_np = np.array(1.+1.j).astype(datatype)

        complex_op = ops.Complex()
        x0 = complex_op(Tensor(x0_np.real, dtype=real_datatype), Tensor(x0_np.imag, dtype=real_datatype))
        y0 = complex_op(Tensor(y0_np.real, dtype=real_datatype), Tensor(y0_np.imag, dtype=real_datatype))
        x1 = complex_op(Tensor(x1_np.real, dtype=real_datatype), Tensor(x1_np.imag, dtype=real_datatype))
        y1 = complex_op(Tensor(y1_np.real, dtype=real_datatype), Tensor(y1_np.imag, dtype=real_datatype))

        out = net(x0, y0).asnumpy()
        expect = np.power(x0_np, y0_np)
        assert np.allclose(out, expect)
        assert out.shape == expect.shape

        out = net(x1, y1).asnumpy()
        expect = np.power(x1_np, y1_np)
        assert np.allclose(out, expect)
        assert out.shape == expect.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pow_dynamic_shape():
    """
    Feature: test dynamic shape feature of Pow op on GPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Net()
    x_dyn = Tensor(shape=[None, 3, 3, None], dtype=ms.float32)
    y_dyn = Tensor(shape=[None, 3, 3, None], dtype=ms.float32)
    net.set_inputs(x_dyn, y_dyn)
    random_shape0 = random.randint(1, 9)
    random_shape1 = random.randint(1, 9)
    x_np = np.random.randint(1, 9, (random_shape0, 3, 3, random_shape1))
    y_np = np.random.randint(1, 9, (random_shape0, 3, 3, random_shape1))
    output = net(Tensor(x_np, ms.float32), Tensor(y_np, ms.float32))
    expect = np.power(x_np, y_np)
    assert np.allclose(output.asnumpy(), expect)
    assert output.asnumpy().shape == expect.shape
