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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)
add1 = P.Add()
mul1 = P.MatMul()
add2 = P.Add()


def add(x, y):
    return add1(x, y)


class Func(nn.Cell):
    def __init__(self):
        super(Func, self).__init__()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()

    def construct(self, x, y):
        init = self.alloc_status()
        sum_ = add(x, y)
        product = mul1(x, y)
        init = F.depend(init, sum_)
        init = F.depend(init, product)
        get_status = self.get_status(init)
        sum_ = F.depend(sum_, get_status)
        product = F.depend(product, get_status)
        out = add2(sum_, product)
        init = F.depend(init, out)
        clear = self.clear_status(init)
        out = F.depend(out, clear)
        return out


grad_s = C.GradOperation(get_all=True, sens_param=True)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
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
        sum2 = add2(sum1, dx[0])
        sum3 = add2(y, dx[1])
        out = add2(sum2, sum3)
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
        super(Net_hyper, self).__init__()
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
        sum2 = add2(sum1[0], dx[0])
        sum3 = add2(sum1[1], dx[1])
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
    class Net(nn.Cell):
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
    net = Net()
    data = net(x)
