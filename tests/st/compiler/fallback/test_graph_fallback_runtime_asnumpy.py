# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test graph JIT Fallback runtime feature """
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore import jit
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


class ConstNet(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())

    def construct(self):
        a = ms.Tensor(np.array(4), ms.int32)
        b = ms.Tensor(np.array(5), ms.int32)
        return self.np_function(a, b)


class Net(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())  # @jit.typing: () -> tensor_type[int32]

    def np_function2(self, a, b):
        x = np.exp(a.asnumpy())
        y = np.exp(b.asnumpy())
        return np.exp(x + y)

    def construct(self, a, b):
        return self.np_function(a, b)


@pytest.mark.skip(reason="Pyexecute output is not any and type is wrong.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_np():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net()(a, b)
    const_output = ConstNet()()
    np.testing.assert_almost_equal(output, const_output, 3)


@pytest.mark.skip(reason="Pyexecute output is not any and type is wrong.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_np_grad():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = ops.grad(Net())(a, b)
    assert output == 0


class Net1(ms.nn.Cell):
    def np_function(self, a, b):
        x = a.asnumpy()
        y = b.asnumpy()
        return np.exp(x + y)

    def construct(self, a, b):
        return self.np_function(a, b)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_np_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net1()(a, b)
    const_output = ConstNet()()
    np.testing.assert_almost_equal(output, const_output, 3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_np_asnumpy_grad():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = ops.grad(Net1())(a, b)
    assert output == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_jit_tensor_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    @ms.jit
    def tensor_asnumpy():
        tensor = ms.Tensor(np.arange(0, 6).reshape(2, 3))
        res = tensor.asnumpy()
        return res

    res = tensor_asnumpy()
    print(res)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_asnumpy_1():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class AsNumpyNet1(ms.nn.Cell):
        def construct(self, x):
            a = ms.Tensor(x.asnumpy())
            b = a.asnumpy()
            return a, b

    net = AsNumpyNet1()
    input_x = ms.Tensor(np.array(10, np.float64))
    out = net(input_x)
    assert out[0].asnumpy() == out[1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_asnumpy_2():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class AsNumpyNet2(ms.nn.Cell):
        def construct(self):
            x = np.arange(15).reshape(3, 5)
            y = Tensor(np.array(x)).asnumpy()
            return y.shape

    net = AsNumpyNet2()
    out = net()
    assert out == (3, 5)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_asnumpy_3():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    @jit
    def test():
        x = Tensor(np.array([1, 2], dtype=np.float32))
        y = x.asnumpy()
        y[0] = 11
        return x, y

    out = test()
    assert (out[0].asnumpy() == out[1]).all()
