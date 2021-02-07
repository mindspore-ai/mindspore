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
""" test outermost net pass non_tensor inputs"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE)


class FirstInputTupleNet(nn.Cell):
    def __init__(self):
        super(FirstInputTupleNet, self).__init__()

    def construct(self, tuple_a, tensor_x, list_b, tensor_y, scalar, dict_c, flag):
        if flag:
            return tensor_x - tuple_a[2] + list_b[1][1]["x"] - tensor_y + scalar - dict_c["x"]
        return tensor_x + tuple_a[2] - list_b[1][1]["y"] + tensor_y - scalar + dict_c["y"]


class GradNet(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_x, list_b, tensor_y, scalar, dict_c, flag):
        return self.grad_all(self.forward_net)(tuple_a, tensor_x, list_b, tensor_y, scalar, dict_c, flag)


class GradNet1(nn.Cell):
    def __init__(self, net, get_all):
        super(GradNet1, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=get_all)

    def construct(self, tuple_a, tensor_x, list_b, tensor_y, tensor_z, dict_c):
        return self.grad_all(self.forward_net)(tuple_a, tensor_x, list_b, tensor_y, tensor_z, dict_c)


x = Tensor(np.ones((2, 2), np.float32))
y = Tensor(np.ones((2, 2), np.float32) * 2)
z = Tensor(np.ones((2, 2), np.float32) * 3)
w = Tensor(np.ones((2, 2), np.float32) * 4)
sl = 6
s = "ok"
arg_t0 = (x, y, z, w)
arg_t1 = (w, y, z, w)
arg_l0 = [[x, x], [[x, y], {"x": x, "y": y, "z": x, "p": y}]]
arg_l1 = [[x, x], [[x, y], {"x": x, "y": y, "z": x, "p": y}]]
args_d0 = {"x": x, "y": y}
args_d1 = {"x": x, "y": y}
flag_0 = True
flag_1 = False


p = Parameter(x, name="weight")
a = np.ones((2, 2))

forward_net = FirstInputTupleNet()
grad_all_inputs_net = GradNet(forward_net, get_all=True)


def test_grad_first_input_net():
    class FirstInputTensorNet(nn.Cell):
        def __init__(self):
            super(FirstInputTensorNet, self).__init__()

        def construct(self, tensor_x, tuple_a, list_b, tensor_y, tensor_z, dict_c):
            return tensor_x + tuple_a[2] - list_b[1][1]["y"] + tensor_y - tensor_z + dict_c["y"]

    grad_fist_input_tensor_net = GradNet1(FirstInputTensorNet(), get_all=False)
    ret = grad_fist_input_tensor_net(z, arg_t0, arg_l0, w, y, args_d0)
    assert np.allclose(ret.asnumpy(), np.ones((2, 2), np.float32))


def test_net_inputs_including_str():
    with pytest.raises(TypeError) as err:
        grad_all_inputs_net(arg_t0, s, arg_l0, w, sl, args_d0, flag_0)
    assert "The inputs types of the outermost network support bool, int, float, tensor, " \
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), " \
           "and tuple or list containing only these types, and dict whose values are these types, " \
           "but got 1th arg is ok" in str(err.value)


def test_outermost_net_pass_parameter():
    with pytest.raises(TypeError) as err:
        forward_net(arg_t0, p, arg_l0, w, sl, args_d0, flag_0)
    assert "The inputs types of the outermost network support bool, int, float, tensor, " \
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), " \
           "and tuple or list containing only these types, and dict whose values are these types, " \
           "but got 1th arg is Parameter (name=weight, shape=(2, 2), dtype=Float32, requires_grad=True)" \
           in str(err.value)


def test_outermost_net_pass_tuple_including_parameter():
    with pytest.raises(TypeError) as err:
        forward_net(arg_t0, z, arg_l0, sl, args_d0, flag_0, (z, w, p))
    assert "The inputs types of the outermost network support bool, int, float, tensor, " \
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), " \
           "and tuple or list containing only these types, and dict whose values are these types, " \
           "but got 6th arg is (" in str(err.value)


def test_outermost_net_pass_list_including_parameter():
    with pytest.raises(TypeError) as err:
        forward_net(arg_t0, z, arg_l0, sl, [z, w, p], args_d0, flag_0)
    assert "The inputs types of the outermost network support bool, int, float, tensor, " \
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), " \
           "and tuple or list containing only these types, and dict whose values are these types, " \
           "but got 4th arg is [" in str(err.value)


def test_grad_net_pass_dict_including_parameter():
    with pytest.raises(TypeError) as err:
        grad_all_inputs_net(arg_t0, z, arg_l0, {"x": z, "y": w, "z": p}, sl, args_d0, flag_0)
    assert "The inputs types of the outermost network support bool, int, float, tensor, " \
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), " \
           "and tuple or list containing only these types, and dict whose values are these types, " \
           "but got 3th arg is {" in str(err.value)
