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
""" test implicit conversion """
import numpy as np
import pytest

from mindspore import Tensor, nn, context, Parameter
from mindspore import dtype as mstype
from mindspore.ops import composite as C


grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


def test_user_define_bprop_check_ok():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float32))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=True)
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, sens)
    assert ret[0].shape == (2, 3)
    assert ret[0].dtype == mstype.float32
    assert (ret[0].asnumpy() == np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], np.float32) * 3).all()


def test_user_define_bprop_no_check_dtype():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float16))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=False)
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, sens)
    assert ret[0].shape == (2, 3)
    assert ret[0].dtype == mstype.float16
    assert (ret[0].asnumpy() == np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], np.float16) * 3).all()


def test_user_define_bprop_check_shape():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2], [2.0, 3.0]], dtype=np.float32))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=True)
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(ValueError) as ex:
        ret = grad_net(x, sens)


def test_user_define_bprop_check_dtype():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float16))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=True)
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(TypeError) as ex:
        ret = grad_net(x, sens)


def test_user_define_bprop_check_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.par = Parameter(Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float32)), name="par")
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float16))

        def construct(self, x):
            ret = x * 2 + self.par
            return ret

        def bprop(self, x, out, dout):
            return dout + x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=True)
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(RuntimeError) as ex:
        ret = grad_net(x, sens)
    assert "When user defines the net bprop, the 'Parameter' data type is not supported in the net." in str(ex.value)


def test_user_define_bprop_check_number():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float32))

        def construct(self, x, y):
            ret = x * 2 + y
            return ret

        def bprop(self, x, y, out, dout):
            return (dout,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y, sens):
            return grad_all_with_sens(self.net)(x, y, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=True)
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(TypeError) as ex:
        ret = grad_net(x, y, sens)
    assert "For user define net bprop, the gradients number: 1 is not equal to the args number: 2." in str(ex.value)
