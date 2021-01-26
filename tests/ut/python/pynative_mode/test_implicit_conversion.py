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
import mindspore as ms

from mindspore import Tensor, nn, Parameter
from mindspore.ops import composite as C
from mindspore.ops import functional as F


grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


def test_float_tensor_and_int_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = 2
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]], dtype=np.float32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_bool_tensor_and_float_add():
    x = Tensor(np.array([[True, False], [False, True]], dtype=np.bool_))
    y = 3.3
    ret_actual = x + y
    ret_expect = Tensor(np.array([[4.3, 3.3], [3.3, 4.3]], dtype=np.float32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_bool_tensor_and_int_add():
    x = Tensor(np.array([[True, False], [False, True]], dtype=np.bool_))
    y = 3
    ret_actual = x + y
    ret_expect = Tensor(np.array([[4, 3], [3, 4]], dtype=np.int64))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_bool_and_int_tensor_add():
    x = True
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_int_tensor_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_float_tensor_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_int_tensor_and_int_tensor_add():
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2, 4, 6], [8, 10, 12]], dtype=np.int32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_bool_tensors_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[True, True, True], [False, False, False]], dtype=np.bool_))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[1.1, 1.2, 1.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_int8_tensor_and_uint8_tensors_add():
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2, 4, 6], [8, 10, 12]], dtype=np.int16))
    assert ret_actual.dtype == ret_expect.dtype
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_str_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = "ok"
    with pytest.raises(TypeError) as er:
        ret = x + y
    assert "For 'Add', the 1th input is a not support implicit conversion type: str" in str(er.value)


def test_float_tensor_and_tuple_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = (1, 2, 3)
    with pytest.raises(TypeError) as er:
        ret = x + y
    assert "For 'Add', the 1th input is a not support implicit conversion type: tuple" in str(er.value)


def test_float_tensor_and_list_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = [1, 2, 3]
    with pytest.raises(TypeError) as er:
        ret = x + y
    assert "For 'Add', the 1th input is a not support implicit conversion type: list" in str(er.value)


def test_float_tensor_and_bool_tensors_add_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            return x + y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y, sens):
            return grad_all_with_sens(self.net)(x, y, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[True, True, True], [False, False, False]], dtype=np.bool_))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, y, sens)
    assert ret[0].dtype == x.dtype
    assert ret[1].dtype == y.dtype
    assert (ret[0].asnumpy() == sens.asnumpy()).all()
    assert (ret[1].asnumpy() == sens.asnumpy().astype(np.bool_)).all()


def test_float_tensor_and_int_tensors_sub_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            return x - y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y, sens):
            return grad_all_with_sens(self.net)(x, y, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, y, sens)
    assert ret[0].dtype == x.dtype
    assert ret[1].dtype == y.dtype
    assert (ret[0].asnumpy() == sens.asnumpy()).all()
    assert (ret[1].asnumpy() == sens.asnumpy() * -1).all()


def test_float16_tensor_and_float32_tensors_sub_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            return x - y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y, sens):
            return grad_all_with_sens(self.net)(x, y, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.int32))
    y = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, y, sens)
    assert ret[0].dtype == x.dtype
    assert ret[1].dtype == y.dtype
    assert (ret[0].asnumpy() == sens.asnumpy()).all()
    assert (ret[1].asnumpy() == sens.asnumpy() * -1).all()


def test_float_tensor_and_int_add_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return x + 2

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, sens)
    assert ret[0].dtype == x.dtype
    assert (ret[0].asnumpy() == sens.asnumpy()).all()


def test_int8_tensor_and_uint8_tensors_add_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            return x + y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y, sens):
            return grad_all_with_sens(self.net)(x, y, sens)

    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))
    sens = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16))
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, y, sens)
    assert ret[0].dtype == x.dtype
    assert ret[1].dtype == y.dtype
    assert (ret[0].asnumpy() == sens.asnumpy()).all()
    assert (ret[1].asnumpy() == sens.asnumpy()).all()

class AssignCheck(nn.Cell):
    """ NetWithNDarray definition """

    def __init__(self):
        super(AssignCheck, self).__init__()
        self.cov_step = Parameter(0.0, name="cov_step", requires_grad=False)

    def construct(self, x, y):
        F.assign(self.cov_step, y)
        F.assign(x, y)
        return x


def test_assign_check_in_sig():
    net = AssignCheck()
    x = Tensor(2, ms.int8)
    y = Tensor(3, ms.uint8)
    with pytest.raises(TypeError) as e:
        net(x, y)
    assert "Parameter" in e.value.args[0]
