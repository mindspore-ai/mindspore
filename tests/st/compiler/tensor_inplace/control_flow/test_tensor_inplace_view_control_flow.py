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
""" test_tensor_inplace_view_control_flow """

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops, mint, lazy_inline
from mindspore.ops.auto_generate.gen_ops_def import select_ext_view_op, expand_dims_view_op, slice_ext_view_op
from mindspore.ops.auto_generate.gen_ops_prim import inplace_copy_op, ExpandDimsView, NarrowView, \
    SelectExtView, SplitTensorView, TransposeView
from mindspore.ops.functional import grad
import mindspore.ops.operations as P
from tests.mark_utils import arg_mark

ms.context.set_context(jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
def test_view_in_control_flow1():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_view_op(input_tensor1, 0, 0)
            else:
                m = select_ext_view_op(input_tensor1, 0, 1)
            m.add_(x)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor(3), Tensor([1, 2]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor(3), Tensor([1, 2]))
        assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_view_in_control_flow2():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_view_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 0)
            else:
                if x < 7:
                    m = select_ext_view_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 1)
            m.add_(x)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor(3), Tensor([[1, 2], [3, 4]]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor(3), Tensor([[1, 2], [3, 4]]))
        assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_view_in_control_flow3():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_view_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 0)
                m.add_(x)
            else:
                if x < 7:
                    m = select_ext_view_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 1)
                m.add_(x)
            m.add_(x)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor(3), Tensor([[1, 2], [3, 4]]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor(3), Tensor([[1, 2], [3, 4]]))
        assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_view_in_control_flow4():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_view_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=1)(Tensor(0), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=1)(Tensor(0), Tensor([1, 2]))
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_view_in_control_flow5():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_view_op(input_tensor1, 0, 0)
            else:
                m = select_ext_view_op(input_tensor1, 0, 1)
            m.add_(x)
            if m > 5:
                n = select_ext_view_op(input_tensor1, 0, 1)
            else:
                n = select_ext_view_op(input_tensor1, 0, 0)
            n.add_(m)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        grad(net, grad_position=1)(Tensor(3), Tensor([1, 2]))
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
def test_view_in_control_flow7():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = foo(input_tensor1)
            m.add_(x)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor(3), Tensor([1, 2]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor(3), Tensor([1, 2]))
        assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_view_in_control_flow8():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]))
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_view_in_control_flow9():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            while 1 < m < 5:
                n = foo(input_tensor1)
                m.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]))
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
def test_view_in_control_flow10():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            for _ in range(10):
                if x < 5:
                    for _ in range(5):
                        x.add_(y)
                        if x >= 5:
                            break
                    m = select_ext_view_op(input_tensor2_1, 0, 0)
                    n = select_ext_view_op(input_tensor2_1, 0, 1)
                else:
                    for _ in range(5):
                        x.add_(y)
                        if x == 1:
                            continue
                    m = select_ext_view_op(input_tensor2_1, 0, 1)
                    n = select_ext_view_op(input_tensor2_1, 0, 0)
                m.add_(x)
                n.add_(y)
            return input_tensor2_1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
        assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_view_in_control_flow10_2():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            if x < 5:
                x.add_(y)
                m = select_ext_view_op(input_tensor2_1, 0, 0)
                n = select_ext_view_op(input_tensor2_1, 0, 1)
            else:
                x.add_(y)
                m = select_ext_view_op(input_tensor2_1, 0, 1)
                n = select_ext_view_op(input_tensor2_1, 0, 0)
            m.add_(x)
            n.add_(y)
            return input_tensor2_1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
        assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_view_in_control_flow11():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            for _ in range(10):
                if x < 5:
                    for _ in range(5):
                        x.add_(y)
                        if x >= 5:
                            break
                    m = foo(input_tensor2_1)
                    n = foo(input_tensor2_1)
                else:
                    for _ in range(5):
                        x.add_(y)
                        if x == 1:
                            continue
                    m = foo(input_tensor2_1)
                    n = foo(input_tensor2_1)
                m.add_(x)
                n.add_(y)
            return input_tensor2_1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=1)(Tensor([3, 4]), Tensor([1, 2]))
        assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_view_in_control_flow12():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            m = foo(input_tensor1_1)
            n = foo(input_tensor2_1)
            while m < 20:
                m.add_(n)
                for _ in range(3):
                    n.add_(m)
            return input_tensor2_1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        grad(net, grad_position=1)(Tensor([1, 2]), Tensor([1, 2]))
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_view_in_control_flow13():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = expand_dims_view_op(input_tensor1, 0)
            else:
                m = expand_dims_view_op(input_tensor1, 1)
            m.add_(m)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        grad(net, grad_position=1)(Tensor(0), Tensor([1, 2]))
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_view_in_control_flow14():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = slice_ext_view_op(input_tensor1, 0, 0, 1, 1)
            else:
                m = slice_ext_view_op(input_tensor1, 0, 0, 1, 2)
            m.add_(m)
            return input_tensor1

    with pytest.raises(RuntimeError) as err:
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        grad(net, grad_position=1)(Tensor(0), Tensor([1, 2]))
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_tensor_view_inplace_grad_with_ctr_flow():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, a, b):
            y = ops.abs(x)
            if mint.equal(a, b):
                y_viewed = select_ext_view_op(y, 0, 0)
            else:
                y_viewed = select_ext_view_op(y, 1, 1)
            inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    with pytest.raises(RuntimeError) as err:
        x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
        x = ms.Tensor(x_np)
        net = Net()
        out_expect = grad(net)(x, x, x)
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net)(x, x, x)
        assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_tensor_view_inplace_grad_with_ctr_flow2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, a, b):
            y = ops.abs(x)
            if mint.equal(a, b):
                y_viewed1 = select_ext_view_op(y, 0, 0)
                y_viewed2 = select_ext_view_op(y, 0, 1)
            else:
                y_viewed1 = select_ext_view_op(y, 1, 0)
                y_viewed2 = select_ext_view_op(y, 1, 1)
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2.add_(2)
            return y

    with pytest.raises(RuntimeError) as err:
        x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
        x = ms.Tensor(x_np)
        net = Net()
        out_expect = grad(net)(x, x, x)
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net)(x, x, x)
        assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_tensor_view_inplace_grad_with_ctr_flow3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, a, b):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            if mint.equal(a, b):
                y_viewed.add_(-2)
            else:
                inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x, x, x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x, x, x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_check4():
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
            if x < y:
                return self.inner_func(y, x)  # pylint: disable=W1114
            return self.inner_func(x, y)

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            return self.func(input_tensor1_1, input_tensor2_1)

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_tensor_view_inplace_grad_control_flow():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            if (x <= y).all():
                y_viewed = slice_ext_view_op(y, 1, 1, 2, 1)
            else:
                y_viewed = y + 1
            inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    with pytest.raises(RuntimeError) as err:
        net = Net()
        x_np = (np.arange(2 * 2 * 2)).reshape((2, 2, 2)).astype(np.float32)
        x = ms.Tensor(x_np)
        out_expect = grad(net)(x)
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net)(x)
        assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())
    assert ("In backpropagation, inplace modification of the output of view operations within control flow is not "
            "supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_control_flow_2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = slice_ext_view_op(y, 1, 1, 2, 1)
            if (x <= y).all():
                inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y_viewed * 2

    net = Net()
    x_np = (np.arange(2 * 2 * 2)).reshape((2, 2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_new_method():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def func(self, input_tensor1_1, input_tensor2_1):
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            if x < x * 2:
                y.mul_(2)
            else:
                y.mul_(3)
            return input_tensor2_1

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            return self.func(input_tensor1_1, input_tensor2_1)

    net = Net()
    out_expect = grad(net, grad_position=1)(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=1)(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_grad_new_method_view():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = slice_ext_view_op(y, 1, 1, 2, 1)
            z_viewed = slice_ext_view_op(y_viewed, 0, 0, 1, 1)
            if (x < x * 2).all():
                inplace_copy_op(z_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y_viewed  # y

    x_np = (np.arange(2 * 2 * 2)).reshape((2, 2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_virtual_view_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            if value < 5:
                y.mul_(value)
            else:
                y.mul(6)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_virtual_view_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor, x, value):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            if value < 5:
                input_tensor1.mul_(2)
            m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_virtual_view_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor, x, value):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            if value < 5:
                input_tensor1.mul_(2)
            m.add_(x)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_virtual_view_case4():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor, x, value):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            if value < 5:
                m.add_(x)
            n = select_ext_view_op(input_tensor1, 1, 0)
            n.add_(x)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_virtual_view_case5():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 0)
            q = select_ext_view_op(n, 0, 0)
            p = select_ext_view_op(q, 0, 0)
            w = select_ext_view_op(p, 0, 0)
            if value < 5:
                n.add_(n)
            return w

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[[[[[-2, -4]]]], [[[[4, 8]]]]]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[[[[[-2, -4]]]], [[[[4, 8]]]]]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_virtual_view_case6():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value1, value2):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            if value2 < 5:
                m.add_(value2)
            else:
                m.add_(value1)
            n = select_ext_view_op(m, 0, 1)
            n.add_(n)
            m.add_(value2)
            if value2 < 5:
                n.add_(n)
            else:
                n.add_(6)
            return n

    with pytest.raises(RuntimeError) as err:
        net = Net()
        out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                     Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                  Tensor([6, 7], dtype=ms.float32), Tensor(2, dtype=ms.float32))
        assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
        assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()
    assert ("In backpropagation, inplace modification of the output of view operations within control "
            "flow is not supported.") in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_virtual_view_case7():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 1)
            u = select_ext_view_op(y, 0, 0)
            v = select_ext_view_op(u, 0, 1)
            if value < 5:
                n.mul_(value)
            else:
                n.add_(value)
            z = v.add(v)
            if value < 5:
                z = z.mul(value)
            return z

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_virtual_view_case8():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 30:
                m.mul_(x)
                input_tensor1.mul_(2)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor(2), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor(2), Tensor([1, 2]))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_view_and_inplace_nested_ctrl_flow_and_lazy_inline():
    """
    Feature: Support tensor inplace view gradient.
    Description: Test control flow, lazy_inline, and the gradients of SplitTensorView, TransposeView, and NarrowView.
    Expectation: Run success.
    """

    class OuterNet(nn.Cell):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            for _ in range(2):
                x = self.block(x, y)
            return x

    class Recursive(nn.Cell):
        @lazy_inline
        def __init__(self):
            super().__init__()
            self.t = 2
            self.transposeview = TransposeView()
            self.split = SplitTensorView()
            self.narrowview = NarrowView()

        def f_ok(self, x, y):
            if P.ReduceSum()(x) < 200:
                x = self.transposeview(x, (1, 0))
                y = self.transposeview(y, (1, 0))
                x.add_(y)
                x = self.narrowview(x, 1, 0, 4)
                x.add_(y)
                if P.ReduceSum()(y) < 200:
                    for _ in range(2):
                        x.add_(y)
                        y.add_(x)
            return x

        def construct(self, x, y):
            out = self.f_ok(x, y)
            return out * self.t

    x_np = np.ones([4, 8]).astype(np.float32)

    y_np = 2 * np.ones([4, 8]).astype(np.float32)

    inner_net = Recursive()
    inner_net.construct = ms.jit(inner_net.construct, backend="ms_backend")
    outer_net = OuterNet(inner_net)
    outer_net.construct = ms.jit(outer_net.construct, backend="ms_backend")
    out = outer_net(Tensor(x_np), Tensor(y_np))
    assert np.allclose(out.asnumpy(), 64 * np.ones([8, 4]).astype(np.float32), 0.0001, 0.0001)
    back_1 = grad(outer_net)(Tensor(x_np), Tensor(y_np))
    back_2 = grad(outer_net, 1)(Tensor(x_np), Tensor(y_np))
    assert np.allclose(back_1.asnumpy(), 8 * np.ones([4, 8]).astype(np.float32), 0.0001, 0.0001)
    assert np.allclose(back_2.asnumpy(), 28 * np.ones([4, 8]).astype(np.float32), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_tensor_view_inplace_grad_control_flow_3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Test if (ternary operation) and for, x->x(view)->view_obj1->view_obj1.inplace, y->y(view)->y.inplace
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.selectview = SelectExtView()
            self.expanddimsview = ExpandDimsView()

        def construct(self, x, y):
            x = ops.abs(x)
            y = ops.abs(y)
            x = self.selectview(x, 0, 1)
            y = self.selectview(y, 0, 1)
            y.add_(2) if P.ReduceSum()(x) < 3 else y.add_(3)  # pylint: disable=W0106
            view_obj1 = None
            for _ in range(2):
                view_obj1 = self.expanddimsview(x, 0)
            view_obj1.mul_(y)
            return view_obj1, y

    x_np = np.ones([4, 8]).astype(np.float32)
    input_x = Tensor(x_np)

    y_np = 2 * np.ones([4, 8]).astype(np.float32)
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
