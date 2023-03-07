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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops.function.math_func import approximate_equal
from mindspore.common.api import _pynative_executor

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
X = np.random.rand(3, 3).astype(np.float32)
Y = np.random.rand(3, 3).astype(np.float32)


class NetApproxmiateEqual(nn.Cell):
    def __init__(self, user_tolerance):
        super(NetApproxmiateEqual, self).__init__()
        self.approximate_equal = P.ApproximateEqual(tolerance=user_tolerance)
        self.x = Parameter(X, name='x')
        self.y = Parameter(Y, name='y')

    def construct(self):
        return self.approximate_equal(self.x, self.y)

    def change_xy(self, x1, x2):
        self.x = Parameter(x1, name='x')
        self.y = Parameter(x2, name='y')
        return self.approximate_equal(self.x, self.y)


def my_approximate_equal(x1, x2, tol=1e-5):
    r"""
    expected output by numpy
    """
    return abs(x1 - x2) < tol


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal():
    """
    Feature: test ops ApproximateEqual.
    Description: Assign random tol, x, y to ApproximateEqual.
    Expectation: match to expected numpy output.
    """
    tol = 0.5
    approx_equal = NetApproxmiateEqual(tol)
    output = approx_equal()
    expect = my_approximate_equal(X, Y, tol)
    assert(output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_docs():
    """
    Feature: test ops ApproximateEqual.
    Description: Tests based on the docs.
    Expectation: match to expected numpy output.
    """
    tol = 2.
    approx_equal = NetApproxmiateEqual(tol)
    x1 = np.array([1, 2, 3]).astype(np.float16)
    x2 = np.array([2, 4, 6]).astype(np.float16)
    output = approx_equal.change_xy(Tensor(x1), Tensor(x2))
    expect = my_approximate_equal(x1, x2, tol)
    assert(output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_different_shape():
    r"""
    Feature: test ops ApproximateEqual.
    Description: error on tolerance is not a float.
    Expectation: catch error, and suggested two parameter should have the same shape.
    """
    with pytest.raises(Exception, match=r"'x1' must have the same shape as 'x2'"):
        tol = 0.5
        approx_equal = NetApproxmiateEqual(tol)
        x1 = np.random.rand(8, 6, 5, 4, 3, 2, 1).astype(np.float32)
        x2 = np.random.rand(7, 6, 5, 4, 3, 2, 1).astype(np.float32)
        approx_equal.change_xy(Tensor(x1), Tensor(x2))
        _pynative_executor.sync()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_tol_not_float():
    r"""
    Feature: test ops ApproximateEqual.
    Description: error on tolerance is not a float.
    Expectation: catch 'TypeError', and suggested the type of 'tolerance' should be 'float'.
    """
    with pytest.raises(TypeError, match=r"the type of 'tolerance' should be 'float'"):
        tol = "str"
        approx_equal = NetApproxmiateEqual(tol)
        x1 = np.random.rand(3, 3, 3).astype(np.float16)
        x2 = np.random.rand(3, 3, 3).astype(np.int)
        approx_equal.change_xy(Tensor(x1), Tensor(x2))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_max_rank():
    r"""
    Feature: test ops ApproximateEqual.
    Description: error on the maximum rank of x is greater or equal to 8.
    Expectation: catch RuntimeError, and suggested the rank of x should be less than 8.
    """
    with pytest.raises(RuntimeError, match=r"the x's rank should be less than 8"):
        tol = 0.5
        approx_equal = NetApproxmiateEqual(tol)
        x1 = np.random.rand(2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2).astype(np.float32)
        x2 = np.random.rand(2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2).astype(np.float32)
        approx_equal.change_xy(Tensor(x1), Tensor(x2))
        _pynative_executor.sync()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_diff_dtype():
    r"""
    Feature: test ops ApproximateEqual.
    Description: take x and y with different precision of float.
    Expectation: Parameters are converted to highest precision data type.
                 Outcome matches to expected numpy output.
    """
    tol = 0.3
    approx_equal = NetApproxmiateEqual(tol)
    x1 = np.random.rand(3, 3, 3).astype(np.float16)
    x2 = np.random.rand(3, 3, 3).astype(np.float32)
    output = approx_equal.change_xy(Tensor(x1), Tensor(x2))
    expect = my_approximate_equal(x1, x2, tol)
    assert(output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_tensor_api():
    r"""
    Feature: test ops ApproximateEqual on tensor API.
    Description: Tests based on the docs and random x, y parameters.
    Expectation: match to expected numpy output.
    """
    tol = 0.5
    output = Tensor.approximate_equal(Tensor(X), Tensor(Y), tol)
    expect = my_approximate_equal(X, Y, tol)
    assert(output.asnumpy() == expect).all()
    x1 = np.random.rand(4, 3, 3).astype(np.float32) / 12345
    x2 = np.random.rand(4, 3, 3).astype(np.float32) / 23456
    output = Tensor(x1).approximate_equal(Tensor(x2))
    expect = my_approximate_equal(x1, x2)
    assert(output.asnumpy() == expect).all()
    tol = 2.
    x1 = Tensor(np.array([1, 2, 3]), mstype.float32)
    x2 = Tensor(np.array([2, 4, 6]), mstype.float32)
    output = Tensor.approximate_equal(Tensor(x1), Tensor(x2), tol)
    expect = my_approximate_equal(x1, x2, tol)
    assert(output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_approxmiate_equal_functional_api():
    r"""
    Feature: test ops ApproximateEqual on functional API.
    Description: Tests based on the docs and random x, y parameters.
    Expectation: match to expected numpy output.
    """
    tol = 0.5
    output = approximate_equal(Tensor(X), Tensor(Y), tol)
    expect = my_approximate_equal(X, Y, tol)
    assert(output.asnumpy() == expect).all()
    x1 = np.random.rand(4, 3, 3).astype(np.float32) / 12345
    x2 = np.random.rand(4, 3, 3).astype(np.float32) / 23456
    output = approximate_equal(Tensor(x1), Tensor(x2))
    expect = my_approximate_equal(x1, x2)
    assert(output.asnumpy() == expect).all()
    tol = 2.
    x1 = Tensor(np.array([1, 2, 3]), mstype.float32)
    x2 = Tensor(np.array([2, 4, 6]), mstype.float32)
    output = approximate_equal(Tensor(x1), Tensor(x2), tol)
    expect = my_approximate_equal(x1, x2, tol)
    assert(output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_approximate_equal():
    """
    Feature: ApproximateEqual cpu op vmap feature.
    Description: test the vmap feature of ApproximateEqual.
    Expectation: success.
    """
    def cal_approximate_equal(input0, input1):
        func = P.ApproximateEqual(2.0)
        return func(input0, input1)

    def manually_batched(func, input0, input1):
        out_manual = []
        for i in range(input0.shape[0]):
            out = func(input0[i], input1[i])
            out_manual.append(out)
        return F.stack(out_manual)

    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]])).astype(np.float32)
    y = Tensor(np.array([[2, 4, 6], [3, 5, 8]])).astype(np.float32)

    out_manual = manually_batched(cal_approximate_equal, x, y)
    out_vmap = F.vmap(cal_approximate_equal, in_axes=(0, 0))(x, y)
    assert np.array_equal(out_manual.asnumpy(), out_vmap.asnumpy())
