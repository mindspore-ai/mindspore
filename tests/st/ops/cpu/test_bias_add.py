# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add = P.BiasAdd()

    def construct(self, x, b):
        return self.bias_add(x, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add4d():
    x_shape = [2, 3, 4, 5]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([0.3, 0.5, 0.7]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add2d():
    x_shape = [2, 3]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([0.3, 0.5, 0.7]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add3d():
    x_shape = [2, 3, 4]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([0.3, 0.5, 0.7]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add5d():
    x_shape = [2, 5, 2, 3, 4]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.bias_add = P.BiasAdd()
        self.mul = P.Mul()
        self.div = P.Div()
        self.add = P.Add()

    def construct(self, x, y, z, w):
        mul_ = self.mul(x, y)
        div_ = self.div(z, w)
        temp = self.bias_add(mul_, div_)
        temp = self.bias_add(temp, div_)
        return self.add(temp, x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net2():
    x_shape = [2, 3, 4]
    x = np.ones(x_shape).astype(np.float32)
    y = np.ones(x_shape).astype(np.float32)
    z = np.array([1.1, 2.2, 3.4]).astype(np.float32)
    w = np.array([10, 10, 10]).astype(np.float32)
    net2 = Net2()
    output = net2(Tensor(x), Tensor(y), Tensor(z), Tensor(w))
    expect_out = (np.array([[[2.22, 2.22, 2.22, 2.22],
                             [2.44, 2.44, 2.44, 2.44],
                             [2.68, 2.68, 2.68, 2.68]],
                            [[2.22, 2.22, 2.22, 2.22],
                             [2.44, 2.44, 2.44, 2.44],
                             [2.68, 2.68, 2.68, 2.68]]]))
    assert np.allclose(output.asnumpy(), expect_out)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_biasadd_vmap():
    """
    Feature: biasadd vmap test on cpu.
    Description: test the rightness of basic biasadd vmap
    Expectation: use vmap rule's result equal to manually batched.
    """

    def cal_biasadd(x, bias):
        return P.BiasAdd(data_format="NCHW")(x, bias)

    vmap_biasadd = vmap(cal_biasadd, in_axes=(0, 0))
    x = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                         [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]).astype(np.float16))
    bias = Tensor(np.array([[0, 1], [1, 2]]).astype(np.float16))
    output = vmap_biasadd(x, bias)
    expect_out = np.array([[[[1, 2],
                             [4, 5]],
                            [[5, 6],
                             [8, 9]]],
                           [[[10, 11],
                             [13, 14]],
                            [[14, 15],
                             [17, 18]]]]).astype(np.float16)
    assert np.allclose(output.asnumpy(), expect_out)


def test_bias_add_forward_functional(nptype):
    """
    Feature: test bias_add forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.arange(6).reshape((2, 3)).astype(nptype))
    bias = Tensor(np.arange(3).reshape((3)).astype(nptype))
    output = F.bias_add(input_x, bias)
    expected = np.array([[0., 2., 4.], [3., 5., 7.]]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add_forward_float32_functional():
    """
    Feature: test bias_add forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_bias_add_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_bias_add_forward_functional(np.float32)


if __name__ == '__main__':
    test_biasadd_vmap()
    test_bias_add_forward_float32_functional()
