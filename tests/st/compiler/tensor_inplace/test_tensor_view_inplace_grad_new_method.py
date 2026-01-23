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
# ==============================================================================
"""Test tensor view inplace grad"""
import pytest
import os
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore import ops, Tensor, nn
from mindspore.nn import ReLU
from mindspore.common.parameter import Parameter
from mindspore import dtype as mstype
from mindspore.ops.auto_generate.gen_ops_prim import (select_ext_view_op, slice_ext_view_op, inplace_copy_op,
                                                      NarrowView, UnstackExtView, BroadcastToView, ExpandDimsView,
                                                      TransposeView)
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_once():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x = ms.Tensor([[0, 1], [2, 3]], dtype=ms.float32)
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_twice():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = slice_ext_view_op(y, 1, 1, 2, 1)
            z_viewed = slice_ext_view_op(y_viewed, 0, 0, 1, 1)
            inplace_copy_op(z_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2 * 2)).reshape((2, 2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_grad():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = slice_ext_view_op(y, 0, 0, 1, 1)
            z = y_viewed1 + 1
            y_viewed2 = slice_ext_view_op(y, 0, 0, 1, 1)
            inplace_copy_op(y_viewed2, z)
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_grad1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2 = select_ext_view_op(y, 0, 1)
            inplace_copy_op(y_viewed2, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_tensor_view_inplace_grad():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            x.add_(y)
            return x

    net = Net()
    out_expect = grad(net)(Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_view_and_inplace_grad_change_same_area1():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            t = select_ext_view_op(m, 0, 0)
            t.add_(x)
            n = select_ext_view_op(input_tensor1, 0, 0)
            z = select_ext_view_op(n, 0, 0)
            z.mul_(2)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_view_and_inplace_grad_change_same_area2():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            t = select_ext_view_op(m, 0, 0)
            t.add_(x)
            input_tensor1.add_(input_tensor1)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_view_and_inplace_grad_change_same_area3():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(m, 0, 0)
            n.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 1]), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 1]), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_tensor_view_inplace_grad_check1():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(x, 0, 1)
            y.add_(2)
            z = x * 2
            z.add_(3)
            return z

    net = Net()
    out_expect = grad(net)(Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_tensor_view_inplace_grad_check2():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            x.add_(4)
            z = x * 2
            z.add_(3)
            return z

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_tensor_view_inplace_grad_check3():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            x.add_(3)
            y.add_(2)
            z = input_tensor1_1 * 2
            z.add_(3)
            return z

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_tensor_view_inplace_grad_check5():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            y.add_(2)
            x.add_(6)
            return x, y

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_check6():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def inner_func(self, x, y):
            y.add_(2)
            z = x * 2
            z.add_(3)
            return z

        def func(self, input_tensor1_1, input_tensor2_1):
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            if x < self.inner_func(x, y):
                return y
            return x

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            return self.func(input_tensor1_1, input_tensor2_1)

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential',
        mem_peak=2048.00)
def test_tensor_view_inplace_grad_check7():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor2, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            x.add_(3)
            z = input_tensor1_1 * 2
            z.add_(3)
            input_tensor2_1.add_(2)
            return input_tensor1_1 + z

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_tensor_view_inplace_grad_check8():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def func(self, input_tensor1):
            x = select_ext_view_op(input_tensor1, 0, 0)
            x.add_(3)
            z = x * 2
            z.add_(3)
            return z

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            y = select_ext_view_op(input_tensor1_1, 0, 0)
            y.add_(2)
            z = self.func(input_tensor1_1)
            if (z > 3).all():
                return input_tensor1_1 + 1
            return z + 1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_setitem_simple_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        @ms.jit(jit_level="O0", backend="ms_backend")
        def construct(self, x, a):
            x[0] = a
            y = x[1][1]  # pylint: disable=unused-variable
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        x = Tensor([[2, 2, 2], [3, 3, 3]])
        a = Tensor([1, 1, 1])
        grad_net = ops.grad(net, grad_position=0)
        x_grad = grad_net(x, a)
        assert np.all(x_grad.asnumpy() == Tensor([[0, 0, 0], [1, 1, 1]]).asnumpy())

        grad_net = ops.grad(net, grad_position=1)
        a_grad = grad_net(x, a)
        assert np.all(a_grad.asnumpy() == Tensor([1, 1, 1]).asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_setitem_simple_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        @ms.jit(jit_level="O0", backend="ms_backend")
        def construct(self, x, a):
            y = x[1]
            y[0] = a
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        x = Tensor([[2, 2, 2], [3, 3, 3]])
        a = Tensor([3])
        grad_net = ops.grad(net, grad_position=0)
        x_grad = grad_net(x, a)
        assert np.all(x_grad.asnumpy() == Tensor([[1, 1, 1], [0, 1, 1]]).asnumpy())

        grad_net = ops.grad(net, grad_position=1)
        a_grad = grad_net(x, a)
        assert np.all(a_grad.asnumpy() == Tensor([1]).asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_setitem_simple_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        @ms.jit(jit_level="O0", backend="ms_backend")
        def construct(self, x, a):
            y = x[1]
            y[0] = a
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        x = Tensor([[2, 2, 2], [3, 3, 3]])
        a = Tensor([3])
        grad_net = ops.grad(net, grad_position=0)
        x_grad = grad_net(x, a)
        assert np.all(x_grad.asnumpy() == Tensor([[0, 0, 0], [0, 1, 1]]).asnumpy())

        grad_net = ops.grad(net, grad_position=1)
        a_grad = grad_net(x, a)
        assert np.all(a_grad.asnumpy() == Tensor([1]).asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_view_inplace_gradient():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: no exception.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            x.add_(y)
            m = select_ext_view_op(input_tensor2_1, 0, 0)
            n = select_ext_view_op(input_tensor2_1, 0, 1)
            m.add_(x)
            n.add_(y)
            return input_tensor2_1

    net = Net()
    out_expect = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@ms.jit(capture_mode='ast', jit_level="O0", backend="ms_backend")
def net_forward(net, *args, **kwargs):
    return net(*args, **kwargs)


@ms.jit(capture_mode='ast', jit_level="O0", backend="ms_backend")
def net_backward(net, *args, **kwargs):
    grad_net = GradOfAllInputs(net)
    return grad_net(*args, **kwargs)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_index_grad():
    """
    Feature: Support tensor index gradient.
    Description: Support tensor index gradient.
    Expectation: no exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.input_x = Parameter(Tensor([[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]], mstype.float32))
            self.relu = ReLU()

        def construct(self, input_y):
            self.input_x[0] //= input_y
            out = self.relu(self.input_x)
            return out

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        ms.set_context(jit_config={"jit_level": "O0"})
        input_me_1 = Tensor([2.0], mstype.float32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        net.set_grad()
        out_me_1 = net_forward(net, input_me_1)
        net_backward(net, input_me_1, out_me_1)
        assert (out_me_1.asnumpy() == [[1, 1, 2, 2], [2, 3, 4, 5]]).all()
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_tensor_index_grad_2():
    """
    Feature: Support tensor index gradient.
    Description: Support tensor index gradient.
    Expectation: no exception.
    """

    def foo(x, y):
        x = ops.abs(x)
        y = ops.abs(y)
        view_obj1 = x[1, 0:2, [0, 1]]
        view_obj1[...] = Tensor(5)
        view_obj2 = y[Tensor(1)]
        view_obj2[None] = 3
        return view_obj2, y

    @ms.jit(backend="ms_backend")
    def foo1(x, y):
        return foo(x, y)

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

        x_np = np.ones([2, 4, 8]).astype(np.float32)
        input_x = Tensor(x_np)
        y_np = 2 * np.ones([2, 4, 8]).astype(np.float32)
        input_y = Tensor(y_np)

        out_forword_expect = foo(input_x, input_y)
        out_back_expect = grad(foo)(input_x, input_y)
        out_back_expect_1 = grad(foo, grad_position=1)(input_x, input_y)
        out_forword_jit = foo1(input_x, input_y)
        out_back_jit = grad(foo1)(input_x, input_y)
        out_back_jit_1 = grad(foo1, grad_position=1)(input_x, input_y)
        assert np.allclose(out_forword_expect[0].asnumpy(), out_forword_jit[0].asnumpy())
        assert np.allclose(out_forword_expect[1].asnumpy(), out_forword_jit[1].asnumpy())
        assert np.allclose(out_back_expect.asnumpy(), out_back_jit.asnumpy())
        assert np.allclose(out_back_expect_1.asnumpy(), out_back_jit_1.asnumpy())

    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_tensor_index_grad_3():
    """
    Feature: Support tensor index gradient.
    Description: Support tensor index gradient.
    Expectation: no exception.
    """

    def foo(x):
        view_obj1 = x[True]
        view_obj2 = view_obj1[None]
        if P.ReduceSum()(x) > 2:
            view_obj2[True] = Tensor(5)
        else:
            view_obj2[True] = Tensor(6)
        return view_obj2, x

    @ms.jit(backend="ms_backend")
    def foo1(x):
        return foo(x)

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

        x_np = np.ones([2, 4]).astype(np.float32)
        input_x = Tensor(x_np)

        out_forword_expect = foo(input_x)
        out_back_expect = grad(foo)(input_x)
        out_forword_jit = foo1(input_x)
        out_back_jit = grad(foo1)(input_x)
        assert np.allclose(out_forword_expect[0].asnumpy(), out_forword_jit[0].asnumpy())
        assert np.allclose(out_forword_expect[1].asnumpy(), out_forword_jit[1].asnumpy())
        assert np.allclose(out_back_expect.asnumpy(), out_back_jit.asnumpy())

    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_tensor_view_inplace_grad_with_tuple_output():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.narrowview = NarrowView()

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            view_obj1 = self.narrowview(y, 1, 0, 4)
            view_obj1.add_(x)
            view_obj2 = self.narrowview(y, 1, 0, 4)
            view_obj2.add_(x)
            return view_obj2, y

    x_np = np.ones([2, 4]).astype(np.float32)
    input_x = Tensor(x_np)
    y_np = 2 * np.ones([2, 4]).astype(np.float32)
    input_y = Tensor(y_np)

    net = Net()
    out_forword_expect = net(input_x, input_y)
    out_back_expect = grad(net)(input_x, input_y)
    out_back_expect_1 = grad(net, 1)(input_x, input_y)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_forword_jit = net(input_x, input_y)
    out_back_jit = grad(net)(input_x, input_y)
    out_back_jit_1 = grad(net, 1)(input_x, input_y)
    assert np.allclose(out_forword_expect[0].asnumpy(), out_forword_jit[0].asnumpy())
    assert np.allclose(out_forword_expect[1].asnumpy(), out_forword_jit[1].asnumpy())
    assert np.allclose(out_back_expect.asnumpy(), out_back_jit.asnumpy())
    assert np.allclose(out_back_expect_1.asnumpy(), out_back_jit_1.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_muti_output_view_inplace_grad():
    """
    Feature: view inplace operation in grad which view operator has multiple outputs.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.unstackview = UnstackExtView()

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            view_obj1 = self.unstackview(x)
            view_obj2 = self.unstackview(y)
            view_obj1 = view_obj1[0]
            view_obj2 = view_obj2[0]
            view_obj1.mul_(y[0])
            return view_obj1, view_obj2

    with pytest.raises(RuntimeError) as err:
        x_np = np.ones([4, 8]).astype(np.float32)
        input_x = Tensor(x_np)
        y_np = 2 * np.ones([4, 8]).astype(np.float32)
        input_y = Tensor(y_np)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        grad(net)(input_x, input_y)
    assert ("In backpropagation, in-place modification operations are not supported for view operators with multiple "
            "outputs.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_view_and_inplace_fallback_pyinterpret():
    """
    Feature: view inplace operation in grad which input of the view operator is PyInterpret node,
    which can be constant folded.
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    def func(x):
        x = ops.abs(x)
        view_obj1 = BroadcastToView()(x, (1, 4, 2))
        view_obj2 = ExpandDimsView()(view_obj1, 0)
        if x[0][0] > 0:
            view_obj2.mul_(2)
        else:
            view_obj2.mul_(3)
        return view_obj2, x

    def func_pyinterpret(x):
        x = TransposeView()(Tensor(np.ones([2, 4]).astype(np.float32)), (1, 0))
        _, y = func(x)
        y.mul_(x)
        return y

    x_np = np.ones([2, 4]).astype(np.float32)
    input_x = Tensor(x_np)
    out_forword_expect = func_pyinterpret(input_x)
    out_back_expect = grad(func_pyinterpret)(input_x)

    func18_jit = ms.jit(func_pyinterpret, backend="ms_backend")
    out_forword_jit = func18_jit(input_x)
    out_back_jit = grad(func18_jit)(input_x)
    assert np.allclose(out_forword_expect.asnumpy(), out_forword_jit.asnumpy())
    assert np.allclose(out_back_expect.asnumpy(), out_back_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_view_and_inplace_while():
    """
    Feature: view inplace operation in grad with while loop
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.transposeview = TransposeView()
            self.reducesum = ops.ReduceSum()
            self.expanddimsview = ExpandDimsView()

        def construct(self, x, y):
            x = ops.Abs()(x)
            y = ops.Abs()(y)
            if self.reducesum(x) < 3 * self.reducesum(y):
                x.add_(y)
                for _ in range(2):
                    while self.reducesum(x) < 200:
                        x = self.transposeview(x, (1, 0))
                        y = self.transposeview(y, (1, 0))
                        x.add_(y)
                        y.add_(y/2)
            else:
                if self.reducesum(x) < 4 * self.reducesum(y):
                    x = self.expanddimsview(x, 1)
                    y = self.expanddimsview(y, 1)
                    x.add_(y)

            if x.shape == (4, 8):
                for _ in range(2):
                    x = self.transposeview(x, (1, 0))
                    y = self.transposeview(y, (1, 0))
                    x.add_(y)
                    y.add_(y/2)

            else:
                x = self.transposeview(x, (1, 0))
                y = self.transposeview(y, (1, 0))
                x.add_(y)

            return x, y

    @ms.jit(backend="ms_backend")
    def grad_under_graph(net, x, y):
        return grad(net, grad_position=(0, 1))(x, y)

    x_np = np.ones([4, 8]).astype(np.float32)
    y_np = 2 * np.ones([4, 8]).astype(np.float32)
    net = Net()
    out_back_expect = grad(net, grad_position=(0, 1))(Tensor(x_np), Tensor(y_np))

    out_back_graph = grad_under_graph(net, Tensor(x_np), Tensor(y_np))
    assert np.allclose(out_back_expect[0].asnumpy(), out_back_graph[0].asnumpy())
    assert np.allclose(out_back_expect[1].asnumpy(), out_back_graph[1].asnumpy())

    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_back_jit = grad(net, grad_position=(0, 1))(Tensor(x_np), Tensor(y_np))
    assert np.allclose(out_back_expect[0].asnumpy(), out_back_jit[0].asnumpy())
    assert np.allclose(out_back_expect[1].asnumpy(), out_back_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_with_tuple_output_case2():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.narrowview = NarrowView()

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            view_obj = self.narrowview(x, 1, 0, 4)
            x.add_(y)
            view_obj.add_(y)
            return view_obj, x

    x_np = np.ones([2, 4]).astype(np.float32)
    input_x = Tensor(x_np)
    y_np = 2 * np.ones([2, 4]).astype(np.float32)
    input_y = Tensor(y_np)
    net = Net()
    out_back_expect = grad(net, grad_position=(0, 1))(input_x, input_y)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_back_jit = grad(net, grad_position=(0, 1))(input_x, input_y)
    assert np.allclose(out_back_expect[0].asnumpy(), out_back_jit[0].asnumpy())
    assert np.allclose(out_back_expect[1].asnumpy(), out_back_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_with_tuple_output_case3():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.expanddimsview = ExpandDimsView()
            self.unstackview = UnstackExtView()

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            view_obj1 = self.expanddimsview(x, 1)
            view_obj2 = self.expanddimsview(y, 1)
            for _ in range(10):
                if ops.ReduceSum()(y) < ops.ReduceSum()(x) * 2:
                    y.div_(2)
                else:
                    y.add_(2)
            if ops.ReduceSum()(x) < 50:
                if ops.ReduceSum()(x) > 40:
                    view_obj2.add_(view_obj1)
                    print("Add print op in branch 1")
                else:
                    print("Add print op in branch 2")
                    view_obj2.add_(view_obj1)
            else:
                print("Add print op in branch 3")
                view_obj2.sub_(view_obj1)
            view_obj1.mul_(y[0])
            return view_obj1, view_obj2

    x_np = np.ones([4, 8]).astype(np.float32)
    input_x = Tensor(x_np)

    y_np = 2 * np.ones([4, 8]).astype(np.float32)
    input_y = Tensor(y_np)

    net = Net()
    out_back_expect = grad(net, grad_position=(0, 1))(input_x, input_y)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_back_jit = grad(net, grad_position=(0, 1))(input_x, input_y)
    assert np.allclose(out_back_expect[0].asnumpy(), out_back_jit[0].asnumpy())
    assert np.allclose(out_back_expect[1].asnumpy(), out_back_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_view_and_inplace_with_inplace_func_call():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            view_obj1 = x[Tensor(1)]
            view_obj2 = y[0:2]
            for _ in range(2):
                if ops.ReduceSum()(x) < ops.ReduceSum()(y) * 2:
                    y[...] = ops.add(y, x)
            view_obj2[1, 0:2] = ops.mul(x[0, 0:2], view_obj1[0:2])
            return view_obj1, view_obj2

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        x_np = np.ones([4, 8]).astype(np.float32)
        y_np = 2 * np.ones([4, 8]).astype(np.float32)
        net = Net()
        out_back_expect = grad(net, grad_position=(0, 1))(Tensor(x_np), Tensor(y_np))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_back_jit = grad(net, grad_position=(0, 1))(Tensor(x_np), Tensor(y_np))
        assert np.allclose(out_back_expect[0].asnumpy(), out_back_jit[0].asnumpy())
        assert np.allclose(out_back_expect[1].asnumpy(), out_back_jit[1].asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_tensor_inplace_grad_with_return_same_out():
    """
    Feature: inplace operation in grad.
    Description: inplace operation in grad.
    Expectation: no exception
    """

    class CtrlForbyIfBR(nn.Cell):
        def __init__(self, t):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()
            self.assignadd = P.AssignAdd()
            self.para = Parameter(t, name="a")

        def construct(self, x, y):
            out = self.add(y, y)
            self.assignadd(self.para, y)
            for _ in range(0, -5, -1):
                x = x - 1
                if x > 0:
                    out = self.mul(out, y)
                else:
                    break
                out = self.add(out, self.para)
            if x > 2:
                return out
            return out

    input_np = np.random.randn(3, 4, 5).astype(np.float32)
    x = Tensor(1)
    t = Tensor(input_np)
    y = Tensor(input_np)
    net = CtrlForbyIfBR(t)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    grad(net)(x, y)
