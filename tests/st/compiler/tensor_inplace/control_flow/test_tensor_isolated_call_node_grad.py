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
""" test_tensor_isolated_call_node_grad.py """
import pytest
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops.auto_generate.gen_ops_def import select_ext_view_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_isolated_switch_call_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    def func(input_tensor, value):
        if value < 5:
            a = select_ext_view_op(input_tensor, 0, 0)
            a.mul_(value)
        return input_tensor

    class Net(nn.Cell):
        def construct(self, input_tensor, value):
            input_tensor1 = ops.abs(input_tensor)
            func(input_tensor1, value)
            return input_tensor1

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
def test_isolated_switch_call_case2():
    """
    Feature: Support tensor inplace gradient.
    Description: Support tensor inplace gradient.
    Expectation: Run success.
    """
    def func(input_tensor, value):
        if value < 5:
            input_tensor.mul_(value)
        return input_tensor

    class Net(nn.Cell):
        def construct(self, input_tensor, value):
            input_tensor1 = ops.abs(input_tensor)
            func(input_tensor1, value)
            return input_tensor1

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
def test_isolated_switch_call_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    def func(input_tensor, value):
        a = select_ext_view_op(input_tensor, 0, 0)
        if value < 5:
            a.mul_(value)
        else:
            a.mul_(2)
        return a

    class Net(nn.Cell):
        def construct(self, input_tensor, value):
            input_tensor1 = ops.abs(input_tensor)
            func(input_tensor1, value)
            return input_tensor1

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
def test_isolated_loop_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    def func(input_tensor, value):
        if value < 5:
            input_tensor.mul_(value)
        return input_tensor

    class Net(nn.Cell):
        def construct(self, input_tensor, value):
            input_tensor1 = ops.abs(input_tensor)
            for _ in range(2):
                func(input_tensor1, value)
            return input_tensor1

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
def test_isolated_loop_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    def func(input_tensor, value):
        if value < 5:
            input_tensor.mul_(value)
        return input_tensor

    class Net(nn.Cell):
        def construct(self, input_tensor, value):
            input_tensor1 = ops.abs(input_tensor)
            while value < 5:
                func(input_tensor1, value)
                value = value + 1
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@pytest.mark.skip(reason="Unsupported now, loop func only last func call node is an isolated node")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_isolated_loop_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    def func(input_tensor, value):
        a = select_ext_view_op(input_tensor, 0, 0)
        if value < 5:
            a.mul_(value)
        return a

    class Net(nn.Cell):
        def construct(self, input_tensor, value):
            input_tensor1 = ops.abs(input_tensor)
            result = input_tensor1
            for _ in range(3):
                result = func(result, value)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()
