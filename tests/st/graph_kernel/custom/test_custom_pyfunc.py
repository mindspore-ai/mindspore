# Copyright 2021 Huawei Technologies Co., Ltd
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

import platform
import numpy as np
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops as ops


def func_single_output(x1, x2):
    return x1 - x2


def func_multi_output(x1, x2):
    return (x1 + x2), (x1 - x2)


output = 0


def func_no_output(x1, x2):
    global output
    output = x1 + x2


class PyFuncNet(nn.Cell):
    def __init__(self, fn, out_shapes, out_types):
        super().__init__()
        self.func = ops.Custom(fn, out_shapes, out_types, "pyfunc")
        self.relu = ops.ReLU()

    def construct(self, x1, x2):
        x = self.func(x1, x2)
        return self.relu(x[0])


def func_with_dtype(ms_dtype, np_dtype):
    shape = (40, 40)
    np.random.seed(42)
    x1 = np.random.randint(-5, 5, size=shape).astype(np_dtype)
    x2 = np.random.randint(-5, 5, size=shape).astype(np_dtype)

    expect = func_single_output(x1, x2)
    expect = ops.ReLU()(Tensor(expect))

    net = PyFuncNet(func_single_output, (shape,), (ms_dtype,))
    x = net(Tensor(x1), Tensor(x2))
    assert np.allclose(x.asnumpy(), expect.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pyfunc_single_output():
    """
    Feature: test case for Custom op with func_type="pyfunc"
    Description: the net runs on GPU while custom pyfunc operator on CPU; GRAPH_MODE; single output
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    func_with_dtype(ms.float16, np.float16)
    func_with_dtype(ms.float32, np.float32)
    func_with_dtype(ms.float64, np.float64)
    func_with_dtype(ms.int32, np.int32)
    func_with_dtype(ms.int64, np.int64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pyfunc_multi_output():
    """
    Feature: test case for Custom op with func_type="pyfunc"
    Description: the net runs on GPU while custom pyfunc operator on CPU; GRAPH_MODE; multiple output
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    shape = (40, 40)
    dtype = ms.float32

    np.random.seed(42)
    x1 = np.random.randint(-5, 5, size=shape).astype(np.float32)
    x2 = np.random.randint(-5, 5, size=shape).astype(np.float32)
    expect, _ = func_multi_output(x1, x2)
    expect = ops.ReLU()(Tensor(expect))

    net = PyFuncNet(func_multi_output, (shape, shape), (dtype, dtype))
    x = net(Tensor(x1), Tensor(x2))

    assert np.allclose(x.asnumpy(), expect.asnumpy())


class PyFuncGraph(nn.Cell):
    def __init__(self, fn, out_shapes, out_types):
        super().__init__()
        self.func = ops.Custom(fn, out_shapes, out_types, "pyfunc")

    def construct(self, x1, x2):
        return self.func(x1, x2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pyfunc_no_output():
    """
    Feature: test case for Custom op with func_type="pyfunc"
    Description:  the net runs on GPU while custom pyfunc operator on CPU; GRAPH_MODE; no output
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    shape = (40, 40)

    np.random.seed(42)
    x1 = np.random.randint(-5, 5, size=shape).astype(np.float32)
    x2 = np.random.randint(-5, 5, size=shape).astype(np.float32)
    func_no_output(x1, x2)
    global output
    expect = output

    net = PyFuncGraph(func_no_output, (), ())
    net(Tensor(x1), Tensor(x2))
    net_output = output

    assert np.allclose(net_output, expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pyfunc_scalar():
    """
    Feature: test case for Custom op with func_type="pyfunc"
    Description:  the net runs on GPU while custom pyfunc operator on CPU; GRAPH_MODE; scalar output
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    shape = ()
    ms_dtype = ms.int32

    x1 = int(10)
    x2 = int(5)
    expect = func_single_output(x1, x2)

    net = PyFuncGraph(func_single_output, shape, ms_dtype)
    x = net(Tensor(x1), Tensor(x2))
    assert np.allclose(x.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pyfunc_pynative():
    """
    Feature: test case for Custom op with func_type="pyfunc"
    Description:  the net runs on CPU; PYNATIVE_MODE
    Expectation: the result match with numpy result
    """
    sys = platform.system()
    if sys != 'Linux':
        pass
    else:
        context.set_context(mode=context.PYNATIVE_MODE)
        shape = (40, 40)

        np.random.seed(42)
        x1 = np.random.randint(-5, 5, size=shape).astype(np.float32)
        x2 = np.random.randint(-5, 5, size=shape).astype(np.float32)
        n1, n2 = func_multi_output(x1, x2)

        net = ops.Custom(func_multi_output, (shape, shape), (ms.float32, ms.float32), "pyfunc")
        out = net(Tensor(x1), Tensor(x2))
        add = ops.Add()
        res = add(out[0], out[1])

        assert np.allclose(res.asnumpy(), n1 + n2)
