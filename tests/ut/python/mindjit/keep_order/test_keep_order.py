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
""" test keep order """
import numpy as np
import mindspore as ms
import mindspore.ops.functional as F
from mindspore import ops, context, nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)
add_op = P.Add()
mul_op = P.MatMul()
add_op2 = P.Add()


def add(x, y):
    return add_op(x, y)


class Func(nn.Cell):
    def __init__(self):
        super().__init__()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()

    def construct(self, x, y):
        init = self.alloc_status()
        sum_ = add(x, y)
        product = mul_op(x, y)
        init = F.depend(init, sum_)
        init = F.depend(init, product)
        get_status = self.get_status(init)
        sum_ = F.depend(sum_, get_status)
        product = F.depend(product, get_status)
        out = add_op2(sum_, product)
        init = F.depend(init, out)
        clear = self.clear_status(init)
        out = F.depend(out, clear)
        return out


grad_s = C.GradOperation(get_all=True, sens_param=True)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = Func()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()

    def construct(self, x, y, sens):
        init = self.alloc_status()
        sum1 = add(x, y)
        dx = grad_s(self.func)(x, y, sens)
        init = F.depend(init, sum1)
        init = F.depend(init, dx)
        get_status = self.get_status(init)
        sum1 = F.depend(sum1, get_status)
        dx = F.depend(dx, get_status)
        sum2 = add_op2(sum1, dx[0])
        sum3 = add_op2(y, dx[1])
        out = add_op2(sum2, sum3)
        init = F.depend(init, out)
        clear = self.clear_status(init)
        out = F.depend(out, clear)
        return out


def test_add():
    x = Tensor(np.ones([3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3]).astype(np.float32))
    func = Func()
    func(x, y)


def test_sens():
    x = Tensor(np.ones([3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3]).astype(np.float32))
    sens = Tensor(np.ones([3, 3]).astype(np.float32))
    net = Net()
    _ = net(x, y, sens)


class Net_hyper(nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = Func()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()

    def construct(self, x, y, sens):
        init = self.alloc_status()
        add1 = add(x, y)
        sum1 = C.hyper_add([add1, add1], [x, y])
        dx = grad_s(self.func)(x, y, sens)
        init = F.depend(init, sum1)
        init = F.depend(init, dx)
        get_status = self.get_status(init)
        sum1 = F.depend(sum1, get_status)
        dx = F.depend(dx, get_status)
        sum2 = add_op2(sum1[0], dx[0])
        sum3 = add_op2(sum1[1], dx[1])
        out = C.hyper_add([sum2, sum2], [sum3, sum3])
        init = F.depend(init, out)
        clear = self.clear_status(init)
        out = F.depend(out, clear)
        return out


def test_hyper_add():
    x = Tensor(np.ones([3, 3]).astype(np.float32))
    y = Tensor(np.ones([3, 3]).astype(np.float32))
    sens = Tensor(np.ones([3, 3]).astype(np.float32))
    net = Net_hyper()
    _ = net(x, y, sens)


def test_keep_order_io_effect_exception_return_dtype():
    class ReturnTypeNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
            self.reduce_sum = P.ReduceSum(keep_dims=True)
            self.dtype = P.DType()
            self.sub = P.Sub()
            self.neg = P.Neg()

        def construct(self, x):
            init = self.alloc_status()
            clear_status = self.clear_status(init)
            x = F.depend(x, clear_status)
            res = self.sub(x, self.neg(x))
            init = F.depend(init, res)
            get_status = self.get_status(init)
            res = F.depend(res, get_status)
            dtype = self.dtype(res)
            return dtype

    value = 655
    data = np.full((8, 5, 3, 1), value, dtype=np.float16)
    x = Tensor(data, dtype=mstype.float16)
    net = ReturnTypeNet()
    data = net(x)


def test_print_effect_trace():
    """
    Feature: Auto Monad
    Description: Test print effect trace.
    Expectation: No exception and no warning.
    """
    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.random.randn(5, 3), ms.float32), name='w')
            self.b = Parameter(Tensor(np.random.randn(3,), ms.float32), name='b')

        def construct(self, x):
            out = ops.matmul(x, self.w)
            print('matmul: ', out)
            out = out + self.b
            print('add bias: ', out)
            return out

    ms.set_context(mode=ms.GRAPH_MODE)
    model = Network()
    x = ops.ones(5, ms.float32)
    model.compile(x)
    out = model(x)
    print('out: ', out)


def test_keep_order_io_effect_base():
    """
    Feature: side effect
    Description: Test io side effect
    Expectation: No exception
    """
    class BaseNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
            self.reduce_sum = P.ReduceSum(keep_dims=True)
            self.sub = P.Sub()
            self.neg = P.Neg()
            self.add_flags(has_effect=True)

        def construct(self, x):
            init = self.alloc_status()
            self.clear_status(init)
            self.sub(x, self.neg(x))
            self.get_status(init)
            flag_sum = self.reduce_sum(init, (0,))
            return flag_sum

    value = 655
    data = np.full((8, 5, 3, 1), value, dtype=np.float16)
    x = Tensor(data, dtype=mstype.float16)
    net = BaseNet()
    net(x)
