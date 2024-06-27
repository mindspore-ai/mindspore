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

import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
import mindspore.ops.operations as P


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_case_1():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(graph_kernel_flags="--enable_low_precision=true --disable_pass=highlevelopt2.atomic_clean")

    class Net1(Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.sub = ops.Sub()
            self.mul = ops.Mul()
            self.sum = ops.ReduceSum(keep_dims=False)
            self.add = ops.Add()
            self.pow = ops.Pow()

        def construct(self, x, y, z):
            t1 = self.sub(x, y)
            t2 = self.mul(t1, x)
            t3 = self.add(y, t2)
            t4 = self.add(t3, t3)
            t5 = z + 1.0
            t6 = self.sum(t4)
            t7 = self.add(t5, t6)
            return t7

    def get_output(x, y, z, net, enable_graph_kernel=False):
        context.set_context(enable_graph_kernel=enable_graph_kernel)
        net_obj = net()
        output = net_obj(x, y, z)
        return output

    N = 8
    x = Tensor(np.random.uniform(1, 2, [N, N, N]).astype(np.float32))
    y = Tensor(np.random.uniform(1, 2, [N, N, N]).astype(np.float32))
    z = Tensor(np.random.uniform(1, 2, [N, N, N]).astype(np.float32))
    expect = get_output(x, y, z, Net1, False)
    output = get_output(x, y, z, Net1, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-2, 1.e-2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_case_2():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(graph_kernel_flags="--enable_low_precision=true")

    class Net2(Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.sqrt = P.Sqrt()
            self.sum = P.ReduceSum(keep_dims=True)
            self.add = P.Add()
            self.neg = P.Neg()

        def construct(self, x, y):
            sqrt_res = self.sqrt(x)
            add_res = self.add(y, sqrt_res)
            neg_res = self.neg(add_res)
            return neg_res

    def get_output(x, y, net, enable_graph_kernel=False):
        context.set_context(enable_graph_kernel=enable_graph_kernel)
        net_obj = net()
        output = net_obj(x, y)
        return output

    N = 16
    x = Tensor(np.random.uniform(1, 2, [N, N]).astype(np.float32))
    y = Tensor(np.random.uniform(1, 2, [N, N]).astype(np.float32))
    expect = get_output(x, y, Net2, False)
    output = get_output(x, y, Net2, True)
    expect_np = expect[0].asnumpy().copy()
    output_np = output[0].asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-2, 1.e-2)
