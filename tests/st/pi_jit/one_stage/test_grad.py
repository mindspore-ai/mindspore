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
import pytest
from tests.mark_utils import arg_mark
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore import dtype as mstype
from mindspore import Tensor, context, Parameter
from mindspore.graph.api import jit, _no_grad
from mindspore.ops.composite import GradOperation
from mindspore.common.parameter import ParameterTuple
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable, get_code_extra

from tests.st.pi_jit.share.utils import match_array, assert_no_graph_break, pi_jit_with_config


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            return ret

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
def test_base_grad_operation_2():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(True, False, False)

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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    assert np.allclose(pynative_res[1].asnumpy(), pijit_res[1].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_3():
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
            self.params = ParameterTuple(self.net.trainable_params())
            self.grad_op = GradOperation(False, True, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net, self.params)(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 1 and len(pijit_res) == 1
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_4():
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
            self.params = ParameterTuple(self.net.trainable_params())
            self.grad_op = GradOperation(True, True, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net, self.params)(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 1 and len(pijit_res[1]) == 1
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_5():
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
            self.params = ParameterTuple(self.net.trainable_params())
            self.sense = Tensor([5, 5, 5])
            self.grad_op = GradOperation(False, False, True)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y, self.sense)
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
def test_base_grad_operation_6():
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
            self.sense = Tensor([5, 5, 5])
            self.params = ParameterTuple(self.net.trainable_params())
            self.grad_op = GradOperation(True, True, True)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net, self.params)(x, y, self.sense)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 1 and len(pijit_res[1]) == 1
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_7():
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
            self.params = self.net.trainable_params()
            self.grad_op = GradOperation(False, True, False)

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net, self.params)(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 1 and len(pijit_res) == 1
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    jit_mode_pi_disable()


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
def test_base_grad_operation_with_keywords_args():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x=x, y=y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = jit(GradNet.construct, capture_mode="bytecode")(grad_net, a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@pytest.mark.skip(reason="pynative handle kwargs failed")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_with_keywords_args_2():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x * y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.sense = Tensor([5, 5, 5])
            self.grad_op = GradOperation(False, False, False)

        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(y=y, x=x)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = jit(GradNet.construct, capture_mode="bytecode")(grad_net, a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_with_vargs():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *args):
            ret = args[0] * args[1]
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.sense = Tensor([5, 5, 5])
            self.grad_op = GradOperation(False, False, False)

        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([[1, 1], [2, 2]])
    b = Tensor([4, 5])
    jit_mode_pi_disable()
    pynative_res = grad_net(a, b)
    jit_mode_pi_enable()
    pijit_res = jit(GradNet.construct, capture_mode="bytecode")(grad_net, a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_functional_grad():
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
            ret = x + y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = ops.grad(self.net)(x, y)
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
def test_functional_grad_2():
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
            ret = x + y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = ops.grad(self.net, grad_position=(0, 1))(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    assert np.allclose(pynative_res[1].asnumpy(), pijit_res[1].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_functional_grad_3():
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
            ret = x + y
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.params = ParameterTuple(self.net.trainable_params())

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = ops.grad(self.net, grad_position=(0, 1), weights=self.params)(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 1 and len(pijit_res[1]) == 1
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_functional_grad_4():
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
            ret = x + y
            return ret, x, y

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.params = ParameterTuple(self.net.trainable_params())

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = ops.grad(self.net, 0, None, has_aux=True)(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 2 and len(pijit_res[1]) == 2
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    assert np.allclose(pynative_res[1][1].asnumpy(), pijit_res[1][1].asnumpy())
    jit_mode_pi_disable()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_functional_grad_5():
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
            ret = x + y
            return ret, x, y

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.params = ParameterTuple(self.net.trainable_params())

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            grad_ret = ops.grad(self.net, 0, None, False, True)(x, y)
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
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert pynative_res[0] == pijit_res[0]
    assert np.allclose(pynative_res[1].asnumpy(), pijit_res[1].asnumpy())
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_no_grad():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Block(nn.Cell):
        def construct(self, x, y):
            return ops.mul(x, y)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def construct(self, x, y):
            with _no_grad():
                a = self.block(x, y)
            b = self.block(x, y)
            return a + b

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    grad_net = GradNet(net)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    o1 = grad_net(a, b)

    net.block.construct = pi_jit_with_config(net.block.construct)
    o2 = grad_net(a, b)

    match_array(o1, o2)
    assert_no_graph_break(net.block.construct, call_count=1)  # call_count=1, should recompile


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_and_grad_operation():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            return ret

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
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    ret = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([3, 3, 3]))
    assert np.all(ret[1].asnumpy() == np.array([1, 1, 1]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_and_grad_operation_with_kwargs():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, *vargs):
            x = vargs[0]
            y = vargs[1]
            ret = x + y
            return ret

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
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    ret = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0].asnumpy() == np.array([3, 3, 3]))
    assert np.all(ret[1].asnumpy() == np.array([1, 1, 1]))


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
    a = Tensor([1,])
    b = Tensor([2,])
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_and_grad_operation_with_side_effect():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            self.a = 1
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
    a = Tensor([1,])
    b = Tensor([2,])
    ret = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0][0].asnumpy() == np.array([3,]))
    assert ret[0][1] == slice(Tensor([1,]), 1, 2)
    assert ret[0][2] is None
    assert ret[0][3] == "a"
    assert net.a == 1


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_and_grad_operation_with_side_effect_2():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            self.a = x - y
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
    a = Tensor([1,])
    b = Tensor([2,])
    ret = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct.__wrapped__)
    assert jcr["break_count_"] == 0
    assert isinstance(ret, tuple)
    assert len(ret) == 2
    assert np.all(ret[0][0].asnumpy() == np.array([3,]))
    assert ret[0][1] == slice(Tensor([1,]), 1, 2)
    assert ret[0][2] is None
    assert ret[0][3] == "a"
    assert np.all(net.a.asnumpy() == np.array([-1,]))


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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_operation_first_input_matches_pynative():
    """
    Feature: ops.GradOperation single input.
    Description: Compare gradients of a basic arithmetic network between pynative and PI JIT.
    Expectation: JIT gradient matches pynative gradient.
    Migrated from: test_pijit_grad.py::test_pijit_grad_operation_first_input
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        @jit(capture_mode="bytecode")
        def construct(self, x):
            y = x + x
            z = x * x
            out = ops.div(y, z)
            return out * self.a

    def build_grad_fn():
        net = Net()

        def grad_fn(x):
            grad_net = ops.GradOperation(False, False, False)(net)
            return grad_net(x)

        return grad_fn

    grad_fn_pynative = build_grad_fn()
    grad_fn_pijit = jit(build_grad_fn(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad = grad_fn_pynative(input_x)
    pijit_grad = grad_fn_pijit(input_x)
    match_array(pynative_grad, pijit_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_operation_all_inputs_stack():
    """
    Feature: ops.GradOperation for all inputs.
    Description: Ensure gradients of stacked numpy/tensor mix match between modes.
    Expectation: Gradients for each input match and remain zeros.
    Migrated from: test_pijit_grad.py::test_pijit_grad_operation_all_inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x, y):
            first, second = x.asnumpy(), y.asnumpy()
            squeezed = np.squeeze(second)
            stacked = np.stack([first, squeezed])
            return Tensor(stacked) * self.a

    def build_grad_fn():
        net = Net()

        def grad_fn(x, y):
            grad_net = ops.GradOperation(True, False, False)(net)
            return grad_net(x, y)

        return grad_fn

    grad_fn_pynative = build_grad_fn()
    grad_fn_pijit = jit(build_grad_fn(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))
    input_y = Tensor(np.ones((2, 1, 3, 1), np.float32))

    pynative_grads = grad_fn_pynative(input_x, input_y)
    pijit_grads = grad_fn_pijit(input_x, input_y)
    for native_grad, jit_grad in zip(pynative_grads, pijit_grads):
        match_array(native_grad, jit_grad)


class Grad(nn.Cell):
    def __init__(self, net, get_all_inputs, get_by_list, sens_param):
        super().__init__()
        self.grad_op = ops.GradOperation(get_all_inputs, get_by_list, sens_param)
        self.net = net

    def construct(self, x):
        grad_net = self.grad_op(self.net, self.net.trainable_params())
        grad = grad_net(x)
        return grad


class GradS(nn.Cell):
    def __init__(self, net, get_all_inputs, get_by_list, sens_param):
        super().__init__()
        self.grad_op = ops.GradOperation(get_all_inputs, get_by_list, sens_param)
        self.net = net

    def construct(self, x, sens):
        grad_net = self.grad_op(self.net, self.net.trainable_params())
        grad = grad_net(x, sens)
        return grad


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_operation_weight_parameter():
    """
    Feature: ops.GradOperation for weights.
    Description: Compare gradients of trainable parameters for numpy-interop network.
    Expectation: Weight gradients match pynative results.
    Migrated from: test_pijit_grad.py::test_pijit_grad_operation_weight
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            x_np = x.asnumpy()
            y = x_np * x_np
            z = Tensor(y)
            out = ops.div(z, z)
            return out * self.a

    def build_grad_net():
        net = Net()
        return Grad(net, False, True, False)

    pynative_grad_net = build_grad_net()
    pijit_grad_net = build_grad_net()
    pijit_grad_net.construct = jit(pijit_grad_net.construct, capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad = pynative_grad_net(input_x)
    pijit_grad = pijit_grad_net(input_x)
    match_array(pynative_grad[0], pijit_grad[0])
    match_array(pynative_grad[0], Tensor(np.ones((2, 3), np.float32)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_operation_sensitivity_parameter():
    """
    Feature: ops.GradOperation with sensitivity.
    Description: Provide explicit sensitivity while comparing gradients between modes.
    Expectation: Gradients match pynative results and scale with the sensitivity.
    Migrated from: test_pijit_grad.py::test_pijit_grad_operation_sens
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')
            self.b = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            inter = (x + x) * self.b
            numpy_out = inter.asnumpy()
            return Tensor(numpy_out) * self.a

    def build_grad_net():
        net = Net()
        return GradS(net, False, True, True)

    pynative_grad_net = build_grad_net()
    pijit_grad_net = build_grad_net()
    pijit_grad_net.construct = jit(pijit_grad_net.construct, capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))
    sense = Tensor(np.ones((2, 3), np.float32) * 2)

    pynative_grad = pynative_grad_net(input_x, sense)
    pijit_grad = pijit_grad_net(input_x, sense)
    match_array(pynative_grad[0], pijit_grad[0])
    match_array(pynative_grad[1], pijit_grad[1])
    match_array(pynative_grad[0], Tensor(np.ones((2, 3), np.float32)) * 4)
    match_array(pynative_grad[1], Tensor(np.zeros((2, 3), np.float32)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_operation_multiple_segments():
    """
    Feature: ops.GradOperation with mixed segments.
    Description: Compare gradients for chained graph/pynative execution with parameters.
    Expectation: Weight gradients match between pynative and PI JIT.
    Migrated from: test_pijit_grad.py::test_pijit_grad_operation_all
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            y = x * x * x
            z = y
            out = ops.div(z, x) * self.a
            return out

    def build_grad_net():
        net = Net()
        return Grad(net, False, True, False)

    pynative_grad_net = build_grad_net()
    pijit_grad_net = build_grad_net()
    pijit_grad_net.construct = jit(pijit_grad_net.construct, capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad = pynative_grad_net(input_x)
    pijit_grad = pijit_grad_net(input_x)
    match_array(pynative_grad[0], pijit_grad[0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_functional_input_gradient():
    """
    Feature: ops.grad for input gradient.
    Description: Compare gradients computed via ops.grad on a mixed graph/pynative cell.
    Expectation: Gradients match and equal the analytical result.
    Migrated from: test_pijit_grad.py::test_pijit_grad_functional_input
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            inter = (x + x) * self.a
            y = inter * inter
            z = y
            out = ops.div(z, inter)
            return out

    def build_grad_fn():
        net = Net()

        def train(x):
            grad_fn = ops.grad(net, grad_position=0, weights=None)
            return grad_fn(x)

        return train

    grad_fn_native = build_grad_fn()
    grad_fn_pijit = jit(build_grad_fn(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad = grad_fn_native(input_x)
    pijit_grad = grad_fn_pijit(input_x)
    match_array(pynative_grad, pijit_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_functional_weights_parameter():
    """
    Feature: ops.grad for weights.
    Description: Compare gradients requested via the weights argument of ops.grad.
    Expectation: Weight gradients match and equal the expected constant tensor.
    Migrated from: test_pijit_grad.py::test_pijit_grad_functional_weights
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            y = x * x * x
            z = y
            out = ops.div(z, x) * self.a
            return out

    def build_grad_fn():
        net = Net()

        def train(x):
            grad_fn = ops.grad(net, grad_position=None, weights=net.a)
            return grad_fn(x)

        return train

    grad_fn_native = build_grad_fn()
    grad_fn_pijit = jit(build_grad_fn(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad = grad_fn_native(input_x)
    pijit_grad = grad_fn_pijit(input_x)
    match_array(pynative_grad, pijit_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_functional_high_order():
    """
    Feature: Higher-order gradients.
    Description: Compare second-order gradients computed via nested GradOperation cells.
    Expectation: Higher-order gradients match pynative expectation.
    Migrated from: test_pijit_grad.py::test_pijit_grad_functional_high_grad
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            y = x * x * x
            z = y
            out = ops.div(z, x) * self.a
            return out

    def build_high_grad():
        net = Net()

        def high_grad(x):
            first_grad = Grad(net, False, False, False)
            second_grad = Grad(first_grad, False, False, False)
            return second_grad(x)

        return high_grad

    grad_fn_native = build_high_grad()
    grad_fn_pijit = jit(build_high_grad(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad = grad_fn_native(input_x)
    pijit_grad = grad_fn_pijit(input_x)
    match_array(pynative_grad, pijit_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_with_aux_for_weights():
    """
    Feature: ops.value_and_grad with has_aux.
    Description: Compare gradients and aux outputs for a cell that returns multiple tensors.
    Expectation: Value/aux structure and gradients match pynative baseline.
    Migrated from: test_pijit_grad.py::test_pijit_grad_functional_has_aux
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            inter = (x + x) * self.a
            y = inter * inter
            z = y
            out = ops.div(z, inter)
            return out, y

    def build_value_and_grad():
        net = Net()

        def train(x):
            grad_net = ops.value_and_grad(net, grad_position=0, weights=net.a, has_aux=True)
            return grad_net(x)

        return train

    train_native = build_value_and_grad()
    train_pijit = jit(build_value_and_grad(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_value, pynative_grad = train_native(input_x)
    pijit_value, pijit_grad = train_pijit(input_x)
    for native_tensor, jit_tensor in zip(pynative_value, pijit_value):
        match_array(native_tensor, jit_tensor)
    for native_tensor, jit_tensor in zip(pynative_grad, pijit_grad):
        match_array(native_tensor, jit_tensor)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_return_ids_for_weights():
    """
    Feature: ops.value_and_grad with return_ids.
    Description: Retrieve gradients by index and parameter id between pynative and PI JIT.
    Expectation: Retrieved gradients match pynative baseline and expected values.
    Migrated from: test_pijit_grad.py::test_pijit_grad_functional_return_ids
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones((2, 3), np.float32)), name='a')

        def construct(self, x):
            inter = (x + x) * self.a
            y = inter * inter
            return ops.div(y, inter)

    def build_value_and_grad():
        net = Net()

        def train(x):
            grad_net = ops.value_and_grad(net, grad_position=0, weights=net.trainable_params(), return_ids=True)
            _, grad = grad_net(x)
            grad0 = ops.get_grad(grad, 0)
            grada = ops.get_grad(grad, net.a)
            return grad0, grada

        return train

    train_native = build_value_and_grad()
    train_pijit = jit(build_value_and_grad(), capture_mode="bytecode")
    input_x = Tensor(np.ones((2, 3), np.float32))

    pynative_grad0, pynative_grada = train_native(input_x)
    pijit_grad0, pijit_grada = train_pijit(input_x)
    match_array(pynative_grad0, pijit_grad0)
    match_array(pynative_grada, pijit_grada)
