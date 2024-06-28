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
import math
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore import dtype as mstype
from mindspore import Tensor, context, Parameter
from mindspore.common.api import jit
from mindspore.ops.composite import GradOperation
from mindspore.common.parameter import ParameterTuple
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_disable, get_code_extra


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    assert np.allclose(pynative_res[1].asnumpy(), pijit_res[1].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 1 and len(pijit_res) == 1
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 1 and len(pijit_res[1]) == 1
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 1 and len(pijit_res[1]) == 1
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0].asnumpy(), pijit_res[0].asnumpy())
    assert np.allclose(pynative_res[1].asnumpy(), pijit_res[1].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 1 and len(pijit_res[1]) == 1
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert np.allclose(pynative_res[0][0].asnumpy(), pijit_res[0][0].asnumpy())
    assert isinstance(pynative_res[1], tuple) and isinstance(pijit_res[1], tuple)
    assert len(pynative_res[1]) == 2 and len(pijit_res[1]) == 2
    assert np.allclose(pynative_res[1][0].asnumpy(), pijit_res[1][0].asnumpy())
    assert np.allclose(pynative_res[1][1].asnumpy(), pijit_res[1][1].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(GradNet.construct, mode="PIJit")
    pijit_res = grad_net(a, b)
    jcr = get_code_extra(GradNet.construct)
    assert jcr["break_count_"] == 0
    assert isinstance(pynative_res, tuple) and isinstance(pijit_res, tuple)
    assert len(pynative_res) == 2 and len(pijit_res) == 2
    assert pynative_res[0] == pijit_res[0]
    assert np.allclose(pynative_res[1].asnumpy(), pijit_res[1].asnumpy())
    jit_mode_pi_disable()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    jit(SecGradNet.construct, mode="PIJit")
    pijit_res = sec_grad_net(a)
    jcr = get_code_extra(SecGradNet.construct)
    assert jcr["break_count_"] == 0
    assert np.allclose(pynative_res.asnumpy(), pijit_res.asnumpy())
    jit_mode_pi_disable()
