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
""" test_grad """
import numpy as np

import mindspore as ms
import mindspore.ops.operations as P
from mindspore import Tensor, context
from mindspore.common.api import ms_function
from mindspore.ops import composite as C
from ...ut_filter import non_graph_engine


# pylint: disable=unused-argument
def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


grad = C.GradOperation()
grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


def mul(x, y):
    return x * y


@ms_function
def mainf(x, y):
    return grad(mul)(x, y)


@non_graph_engine
def test_grad():
    mainf(1, 2)


@non_graph_engine
def Xtest_expand_dims_grad():
    """ test_expand_dims_grad """
    input_tensor = Tensor(np.array([[2, 2], [2, 2]]))
    expand_dims = P.ExpandDims()

    def fn(x):
        output = expand_dims(x, 0)
        return output

    out = fn(input_tensor)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()))
    args = [input_tensor, sens]
    gout = gfn(*args)
    expect = np.ones([2, 2])
    assert np.all(gout[0].asnumpy() == expect)


def test_cast_grad():
    """ test_cast_grad """
    input_np = np.random.randn(2, 3).astype(np.float32)
    input_x = Tensor(input_np)

    td = ms.int32
    cast = P.Cast()

    def fn(x):
        output = cast(x, td)
        return output

    out = fn(input_x)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()))
    args = [input_x, sens]
    gout = gfn(*args)
    expect = np.ones((2, 3), dtype=np.float32)
    assert np.all(gout[0].asnumpy() == expect)


@non_graph_engine
def test_reshape_grad():
    """ test_reshape_grad """
    input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]))
    shp = (3, 2)
    reshape = P.Reshape()

    def fn(x):
        output = reshape(x, shp)
        return output

    out = fn(input_tensor)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()))
    args = [input_tensor, sens]
    gout = gfn(*args)
    expect = np.ones([2, 3])
    assert np.all(gout[0].asnumpy() == expect)


def test_transpose_grad():
    """ test_transpose_grad """
    input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    perm = (0, 2, 1)
    transpose = P.Transpose()

    def fn(x):
        output = transpose(x, perm)
        return output

    out = fn(input_tensor)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()))
    args = [input_tensor, sens]
    gout = gfn(*args)
    expect = np.ones([2, 2, 3])
    assert np.all(gout[0].asnumpy() == expect)


def test_select_grad():
    """ test_select_grad """
    select = P.Select()
    cond = Tensor(np.array([[True, False, False], [False, True, True]]))
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))
    y = Tensor(np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32))

    def fn(cond, x, y):
        output = select(cond, x, y)
        return output

    out = fn(cond, x, y)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()).astype(np.float32))
    args = [cond, x, y, sens]
    gout = gfn(*args)
    expect_cond = np.zeros_like(cond.asnumpy())
    expect_x = np.array([[1, 0, 0], [0, 1, 1]])
    expect_y = np.array([[0, 1, 1], [1, 0, 0]])
    assert np.all(gout[0].asnumpy() == expect_cond)
    assert np.all(gout[1].asnumpy() == expect_x)
    assert np.all(gout[2].asnumpy() == expect_y)


@non_graph_engine
def test_squeeze_grad():
    """ test_squeeze_grad """
    input_tensor = Tensor(np.ones(shape=[3, 2, 1]))
    squeeze = P.Squeeze(2)

    def fn(x):
        output = squeeze(x)
        return output

    out = fn(input_tensor)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()))
    args = [input_tensor, sens]
    gout = gfn(*args)
    expect = np.ones([3, 2, 1])
    assert np.all(gout[0].asnumpy() == expect)


def test_SubGrad():
    """ test_SubGrad """
    input_x = Tensor(np.array([[2, 2]]))
    input_y = Tensor(np.array([[2, 2], [2, 2]]))
    sub = P.Sub()

    def fn(x, y):
        output = sub(x, y)
        return output

    out = fn(input_x, input_y)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()))
    args = [input_x, input_y, sens]
    gout = gfn(*args)
    expect_dx = np.ones([1, 2]).astype(np.int32) * 2  # reduce sum dout to the shape of x
    expect_dy = np.ones([2, 2]).astype(np.int32) * (-1)
    assert np.array_equal(gout[0].asnumpy(), expect_dx)
    assert np.array_equal(gout[1].asnumpy(), expect_dy)


def test_MulGrad():
    """ test_MulGrad """
    input_x = Tensor(np.array([[2, 2], [2, 2]], np.float32))
    input_y = Tensor(np.array([[3, 3], [3, 3]], np.float32))
    mymul = P.Mul()

    def fn(x, y):
        output = mymul(x, y)
        return output

    out = fn(input_x, input_y)
    gfn = grad_all_with_sens(fn)
    sens = Tensor(np.ones_like(out.asnumpy()) * 3)
    args = [input_x, input_y, sens]
    gout = gfn(*args)
    expect_dx = np.ones([2, 2], np.float32) * 9
    expect_dy = np.ones([2, 2], np.float32) * 6
    assert np.all(gout[0].asnumpy().shape == expect_dx.shape)
    assert np.all(gout[0].asnumpy() == expect_dx)
    assert np.all(gout[1].asnumpy().shape == expect_dy.shape)
    assert np.all(gout[1].asnumpy() == expect_dy)
