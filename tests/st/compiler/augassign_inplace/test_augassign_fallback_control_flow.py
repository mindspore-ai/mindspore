# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test augassign fallback control flow."""

import pytest
import numpy as np
from mindspore import nn
import mindspore as ms
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.ops.functional import grad
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_assignment_fallback_1():
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace with fallback feature.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = 6

        def construct(self, x, y, i):
            x = ops.abs(x)
            y = ops.abs(y)
            i = ops.abs(i)
            while self.a < 10:
                self.a *= 2
            return x, y, i


    i = 3 * np.ones([2, 3], dtype=np.float32)
    x = 5 * np.ones([2, 3], dtype=np.float32)
    y = np.ones([2, 3], dtype=np.float32)
    net = Net()
    pynative_out = grad(net, (0, 1, 2))(Tensor(x), Tensor(y), Tensor(i))
    net.construct = ms.jit(net.construct, backend='ms_backend')
    jit_out = grad(net, (0, 1, 2))(Tensor(x), Tensor(y), Tensor(i))
    for i in range(2):
        assert np.allclose(pynative_out[i].asnumpy(), jit_out[i].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_assignment_fallback_2():
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace with fallback feature.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reducesum = P.ReduceSum()
            self.a = 6

        def construct(self, x, y, i):
            x = ops.abs(x)
            y = ops.abs(y)
            i = ops.abs(i)
            if self.reducesum(i) > 1:
                if self.reducesum(i) < 50:
                    x += y
                else:
                    x -= y
                for _ in range(2):
                    x /= y
                    x *= i
            else:
                x *= (y + 2)

            while self.a < 10:
                y //= i
                x //= i
                self.a *= 2
            while self.a > 10:
                i *= Tensor(2)
                self.a /= 2
                self.a -= 2
            return x, y, i

    with pytest.raises(RuntimeError) as raise_info:
        i = 3 * np.ones([2, 3], dtype=np.float32)
        x = 5 * np.ones([2, 3], dtype=np.float32)
        y = np.ones([2, 3], dtype=np.float32)

        net = Net()
        out = net(Tensor(x), Tensor(y), Tensor(i))
        out_back = grad(net, (0, 1, 2))(Tensor(x), Tensor(y), Tensor(i))

        net.construct = ms.jit(net.construct, backend='ms_backend')
        out_jit = net(Tensor(x), Tensor(y), Tensor(i))
        out_back_jit = grad(net, (0, 1, 2))(Tensor(x), Tensor(y), Tensor(i))

        for i in range(3):
            assert np.allclose(out[i].asnumpy(), out_jit[i].asnumpy(), 0.00001, 0.00001)
        for i in range(3):
            assert np.allclose(out_back[i].asnumpy(), out_back_jit[i].asnumpy(), 0.00001, 0.00001)
    assert "Unsupported output: tuple output with dynamic len is not supported in JIT fallback" in str(raise_info.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_assignment_fallback_3():
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace with fallback feature.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = ms.Parameter(Tensor(np.random.randn(2, 3), ms.float32), name='a')
            self.c = 44
            self.reducesum = P.ReduceSum()

        def func(self, x):
            while self.c < 50:
                x += self.a
                x -= self.a
                self.c += 1

        def func1(self, y):
            while self.c > 20:
                y *= self.a
                y /= self.a
                self.c /= 2

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            m = 6
            n = 6
            x += True
            y -= False
            x %= y
            while m < 10:
                while n > 5:
                    if self.reducesum(x) > 1:
                        self.func(x)
                    else:
                        print("aaaaa")

                    for _ in range(2):
                        self.func1(y)

                    n -= 1
                m *= 2

            return x * y

    x = 5 * np.ones([2, 3], dtype=np.float32)
    y = np.ones([2, 3], dtype=np.float32)

    net = Net()
    out = net(Tensor(x), Tensor(y))
    out_back = grad(net, (0, 1))(Tensor(x), Tensor(y))
    out_back_a = grad(net, None, net.a)(Tensor(x), Tensor(y))

    net.construct = ms.jit(net.construct, backend='ms_backend')
    out_jit = net(Tensor(x), Tensor(y))
    out_back_jit = grad(net, (0, 1))(Tensor(x), Tensor(y))
    out_back_jit_a = grad(net, None, net.a)(Tensor(x), Tensor(y))

    for i in range(2):
        assert np.allclose(out[i].asnumpy(), out_jit[i].asnumpy(), 0.00001, 0.00001)
    for i in range(2):
        assert np.allclose(out_back[i].asnumpy(), out_back_jit[i].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(out_back_a.asnumpy(), out_back_jit_a.asnumpy(), 0.00001, 0.00001)
