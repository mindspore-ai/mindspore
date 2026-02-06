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
import numpy as np
import mindspore as ms
from mindspore import context

ms.set_context(mode=ms.GRAPH_MODE)
context.set_context(jit_level='O0')


class ConstNet(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())

    def construct(self):
        a = ms.Tensor(np.array(4), ms.int32)
        b = ms.Tensor(np.array(5), ms.int32)
        return self.np_function(a, b)


def test_fallback_np():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """

    class Net(ms.nn.Cell):
        def np_function(self, a, b):
            return np.exp(a.asnumpy() + b.asnumpy())

        def np_function2(self, a, b):
            x = np.exp(a.asnumpy())
            y = np.exp(b.asnumpy())
            return np.exp(x + y)

        def construct(self, a, b):
            return self.np_function(a, b)

    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net()(a, b)
    const_output = ConstNet()()
    np.testing.assert_almost_equal(output, const_output, 3)


def test_fallback_np_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    class Net1(ms.nn.Cell):
        def np_function(self, a, b):
            x = a.asnumpy()
            y = b.asnumpy()
            return np.exp(x + y)

        def construct(self, a, b):
            return self.np_function(a, b)

    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net1()(a, b)
    const_output = ConstNet()()
    np.testing.assert_almost_equal(output, const_output, 3)


def test_tensor_asnumpy():
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


def test_jit_tensor_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    def tensor_asnumpy():
        tensor = ms.Tensor(np.arange(0, 6).reshape(2, 3))
        res = tensor.asnumpy()
        return res

    res = tensor_asnumpy()
    print(res)
