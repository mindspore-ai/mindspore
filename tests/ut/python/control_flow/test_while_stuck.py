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
import mindspore as ms
from mindspore import jit, mutable, Tensor, vmap, ops
from mindspore.nn import Cell
import numpy as np


def test_mutable_scalar_in_while():
    """
    Feature: Set Constants Scalar as mutable.
    Description: Check whether scalar is broaden.
    Expectation: while won't stuck in infer process.
    """

    @jit
    def mutable_scalar_while():
        i = mutable(1)
        j = Tensor([1])
        while j < 10:
            j = j + 2
            i = i + 2
        return j

    ms.context.set_context(precompile_only=True)
    mutable_scalar_while()


def test_mutable_scalar_in_while_grad():
    """
    Feature: Set Constants Scalar as mutable.
    Description: Check whether scalar is broaden.
    Expectation: while won't stuck in grad process.
    """

    @jit
    def mutable_scalar_while(x):
        i = mutable(1)
        j = Tensor([1])
        while j < 10:
            j = j + x
            i = i + 2
        return j

    @jit
    def mutable_scalar_while_grad(x):
        return ms.ops.grad(mutable_scalar_while)(x)

    ms.context.set_context(precompile_only=True)
    mutable_scalar_while_grad(Tensor([1]))


def test_auto_broaden_scalar_in_while():
    """
    Feature: Broaden scalar arg if it is an arg of no-expanding while header.
    Description: Check whether scalar is broaden.
    Expectation: while won't stuck in infer process.
    """

    @jit
    def scalar_add_in_while():
        i = 1
        j = Tensor([1])
        while j < 10:
            j = j + 2
            i = i + 2
        return j

    ms.context.set_context(precompile_only=True)
    scalar_add_in_while()


def test_auto_broaden_scalar_in_while_grad():
    """
    Feature: Broaden scalar arg if it is an arg of no-expanding while header.
    Description: Check whether scalar is broaden.
    Expectation: while won't stuck in grad process.
    """

    @jit
    def scalar_add_in_while(x):
        i = 1
        j = Tensor([1])
        while j < 10:
            j = j + x
            i = i + 2
        return j

    @jit
    def scalar_add_in_while_grad(x):
        return ms.ops.grad(scalar_add_in_while)(x)

    ms.context.set_context(precompile_only=True)
    scalar_add_in_while_grad(Tensor([1]))


def test_auto_broaden_scalar_in_while_grad_grad():
    """
    Feature: Broaden scalar arg if it is an arg of no-expanding while header.
    Description: Check whether scalar is broaden.
    Expectation: while won't stuck in grad process.
    """

    @jit
    def scalar_add_in_while(x):
        i = 1
        j = Tensor([1])
        while j < 10:
            j = j + x
            i = i + 2
        return j

    @jit
    def scalar_add_in_while_grad_grad(x):
        return ms.ops.grad(ms.ops.grad(scalar_add_in_while))(x)

    ms.context.set_context(precompile_only=True)
    scalar_add_in_while_grad_grad(Tensor([1]))


def test_recursive_func():
    """
    Feature: Recursive func test.
    Description: For recursive func, x is a Constant Tensor, because IGNORE_VALUE flag can't be set to recursive_func,
                 so arg x won't be broaden, the recursive func will be infer endlss. Now this case can pass, because
                 the IGNORE_VALUE are wrongly attached to 'if-switch' graph, which is recursive_func by coincidence.
    Expectation: while won't stuck in infer process.
    """

    def recursive_func(x, max_num):
        if x > max_num:
            return x
        x = x + 1
        return recursive_func(x, max_num)

    @jit
    def test_net(max_num):
        init_x = Tensor([0])
        return recursive_func(init_x, max_num)

    ms.context.set_context(precompile_only=True)
    test_net(Tensor(2))


def test_vmap_while():
    """
    Feature: Vmap with control flow.
    Description: In the situation of vamp with while, axis is a const scalar and as arg of while header, can't broaden
                 this arg when header condition is variable.
    Expectation: No exception raised.
    """

    class Net(Cell):
        @jit
        def construct(self, x, y):
            out = y
            while ops.less(x, 2):
                out = ops.add(y, out)
                x = ops.add(x, 1)
            return out

    net = Net()
    x = Tensor([0], ms.dtype.float32)
    y = Tensor(np.ones([3, 4]), ms.dtype.float32)
    ms.context.set_context(precompile_only=True)
    vmap(net, in_axes=(None, 1), out_axes=1)(x, y)
