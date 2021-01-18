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
from dataclasses import dataclass

import mindspore.nn as nn
from mindspore.ops import Primitive
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants

scala_add = Primitive(Constants.kScalarAdd)


@dataclass
class InferFoo:
    x: int
    y: int

    def inf_add(self):
        return scala_add(self.x, self.y)


def test_dataclass_fun_add(x):
    foo = InferFoo(x, 1.2)
    return foo.inf_add()


@dataclass
class ResolveFoo:
    x: float
    y: float

    def inf_sub(self):
        return self.x - self.y


def test_dataclass_fun_sub(x):
    foo = ResolveFoo(x, 0.1)
    return foo.inf_sub()


def test_fun_sub(x, y):
    return x - y


def test_fun_add(x, y):
    return scala_add(x, y)


class AddNet(nn.Cell):
    def __init__(self):
        super(AddNet, self).__init__()

    def construct(self, x, y):
        return F.scalar_add(x, y)

    def get_params(self):
        return None


def test_net_construct_add():
    model = AddNet()
    return model


class SubNet(nn.Cell):
    def __init__(self):
        super(SubNet, self).__init__()

    def construct(self, x, y):
        return F.scalar_sub(x, y)

    def get_params(self):
        return None


def test_net_construct_sub():
    model = SubNet()
    return model


def test_infer_for(xs, y):
    rval = y
    for x in xs:
        rval = rval + x
    return rval


def test_multiple_conv2d():
    conv1 = P.Conv2D(out_channel=2, pad_mode="pad", kernel_size=(5, 5), pad=0, stride=2, dilation=1)
    conv2 = P.Conv2D(out_channel=2, pad_mode="pad", kernel_size=(5, 5), pad=1, stride=2, dilation=1)

    def get_conv(x, w1, w2):
        return conv2(conv1(x, w1), w2)

    return get_conv


def test_graph_infer_defaults():
    def func_call(x, y, p=10, q=20):
        return x - y - p - q

    def test_call_variable():
        x = 100
        y = 20
        return func_call(x, y)

    return test_call_variable


def test_graph_infer_vararg_0():
    def func_call(x, y, z):
        return x - y - z

    def test_call_variable():
        args = (6, 2, 3)
        return func_call(*args)

    return test_call_variable


def test_graph_infer_vararg():
    def func_call(x, y, *args):
        return x - y - args[0]

    def test_call_variable():
        x = 30
        y = 20
        args = (1, 2, 3)
        return func_call(x, y, *args)

    return test_call_variable


def test_graph_infer_vararg_kwonlyargs():
    def func_call(x, y, *args, p, q):
        return x - y - args[1] - p - q

    def test_call_variable():
        x = 100
        y = 20
        args = (1, 2, 3)
        p = 10
        q = 20
        return func_call(x, y, *args, p=p, q=q)

    return test_call_variable


def test_graph_infer_kwarg():
    def func_call(x, y, **kwargs):
        return x - y - kwargs["a"]

    def test_call_variable():
        x = 30
        y = 20
        kwargs = {"a": 3, "b": 2}
        return func_call(x, y, **kwargs)

    return test_call_variable


def test_graph_infer_vararg_kwonlyargs_kwarg():
    def func_call(x, y, *args, p, q, **kwargs):
        return x - y - args[2] - p - q - kwargs["z"]

    def test_call_variable():
        x = 100
        y = 20
        args = (1, 2, 3)
        kwargs = {"a": 3, "b": 2}
        p = 10
        q = 20
        return func_call(x, y, *args, p=p, q=q, z=1, **kwargs)

    return test_call_variable


def test_graph_infer_vararg_kwonlyargs_kwarg_defaults():
    def func_call(x, y, *args, p=1, q=2, **kwargs):
        return x - y - args[0] - p - q - kwargs["z"]

    def test_call_variable():
        x = 100
        y = 20
        args = (1, 2, 3)
        kwargs = {"a": 1, "b": 2}
        p = 10
        q = 20
        return func_call(x, y, *args, p=p, z=10, m=q, **kwargs)

    return test_call_variable
