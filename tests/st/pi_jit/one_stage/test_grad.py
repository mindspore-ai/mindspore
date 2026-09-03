# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test basic operation with one stage"""
from tests.mark_utils import arg_mark
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore import dtype as mstype
from mindspore import Tensor, context, Parameter
from mindspore.common.api import jit
from mindspore.ops.composite import GradOperation
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable, get_code_extra

from tests.st.pi_jit.share.utils import match_array


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_8():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([1]), name='w')

        def construct(self, x, y):
            ret = self.w * x + y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, True, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net, self.net.trainable_params())(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 1
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 1 and len(pijit_res) == 1
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_second_grad_operation():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x):
            ret = ops.sin(x)
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        def construct(self, x):
            grad_ret = self.grad_op(self.net)(x)
            return grad_ret

    class SecGradNet(nn.Cell):
        def __init__(self, net, ):
            super(SecGradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        @jit(capture_mode="bytecode")
        def construct(self, x):
            grad_ret = self.grad_op(self.net)(x)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    sec_grad_net = SecGradNet(grad_net)
    a = Tensor([1, 1, 1], dtype=mstype.float32)
    jit_mode_pi_disable()
    pynative_res = sec_grad_net(a)
    jit_mode_pi_enable()
    pijit_res = sec_grad_net(a)
    jcr = get_code_extra(SecGradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_with_invalid_output():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            return ret, "a", slice(x, 1, 2)

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_with_invalid_output_2():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *args):
            ret = args[0] + args[1]
            return ret, "a", slice(args[0], 1, 2), {"1": args[0]}

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_with_invalid_output_3():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *args, **kwargs):
            ret = args[0] + args[1]
            return ret, "a", slice(args[0], 1, 2), {"1": args[0]}

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_and_grad_operation_with_invalid_output():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            return ret, slice(x, 1, 2), None, "a"

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = ops.value_and_grad(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1.,])
    b = Tensor([2.,])
    ret = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0][0].asnumpy() == np.array([3,]))
    assert ret[0][1] == slice(Tensor([1,]), 1, 2)
    assert ret[0][2] is None
    assert ret[0][3] == "a"
    assert np.all(ret[1].asnumpy() == np.array([1,]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_grad_operation_and_has_graph_break():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with graph break
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        @jit(capture_mode='bytecode')
        def construct(self, x: Tensor):
            y = x + x
            z = x * x
            out = ops.div(y, z)
            return out * self.a

    model = Model()

    def fn(x: Tensor):
        m = ops.GradOperation(False, False, False)(model)
        return m(x)

    a = Tensor(np.ones((2, 3), np.float32))
    o1 = fn(a)

    compiled_fn = jit(fn, capture_mode='bytecode')
    a = Tensor(np.ones((2, 3), np.float32))
    o2 = compiled_fn(a)

    match_array(o1, o2)
