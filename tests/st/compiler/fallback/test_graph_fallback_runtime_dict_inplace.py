# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test graph dict inplace operation"""
import pytest
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import jit, jit_class, nn, ops
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)

global_dict_1 = {"1": 1}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_global_dict_used_in_graph():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo():
        return global_dict_1

    res = foo()
    assert id(res) == id(global_dict_1)


global_dict_2 = {"1": [1, 2, 3, 4]}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_dict_used_in_graph_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo():
        return global_dict_2["1"]

    res = foo()
    assert id(res) == id(global_dict_2["1"])


global_dict_3 = {"1": ([np.array([1, 2, 3]), np.array([4, 5, 6])], "test")}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_dict_used_in_graph_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo():
        return global_dict_3["1"]

    res = foo()
    assert id(res[0]) == id(global_dict_3["1"][0])


global_input_dict_1 = {"1": 1}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_global_dict_as_graph_input():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo(dict_input):
        return dict_input

    res = foo(global_input_dict_1)
    assert id(res) == id(global_input_dict_1)


global_input_dict_2 = {"1": [1, 2, 3, 4]}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_dict_as_graph_input_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo(dict_input):
        return dict_input["1"]

    res = foo(global_input_dict_2)
    assert id(res) == id(global_input_dict_2["1"])


global_input_dict_3 = {"1": ([1, 2, 3, 4], 5, 6)}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_dict_as_graph_input_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo(dict_input):
        return dict_input["1"]

    res = foo(global_input_dict_3)
    assert id(res[0]) == id(global_input_dict_3["1"][0])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_inplace_with_attribute():
    """
    Feature: Enable dict do inplace operation.
    Description: support dict inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x

    x = {"1": 1, "2": 2}
    net = Net(x)
    ret = net()
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_inplace_with_attribute_2():
    """
    Feature: Enable dict do inplace operation.
    Description: support dict inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x["2"]

    x = {"1": 1, "2": [1, 2, 3, 4]}
    net = Net(x)
    ret = net()
    assert id(x["2"]) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_inplace_setitem():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo(dict_input):
        dict_input["a"] = 3
        return dict_input

    x = {"a": 1, "b": 2}
    res = foo(x)
    assert res == {"a": 3, "b": 2}
    assert id(x) == id(res)


@pytest.mark.skip(reason="Dictionary with no return will be convert to tuple")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_inplace_setitem_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo(dict_input):
        dict_input["a"] = 3

    x = {"a": 1, "b": 2}
    foo(x)
    assert x == {"a": 3, "b": 2}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inplace_setitem_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit
    def foo(dict_input, list_input):
        dict_input["b"] = list_input
        return dict_input

    x = {"a": 1, "b": 2}
    y = [1, 2, 3, 4]
    res = foo(x, y)
    assert res == {"a": 1, "b": [1, 2, 3, 4]}
    assert id(x) == id(res)
    assert id(y) == id(res["b"])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inplace_setitem_with_attribute():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x["1"] = 10
            self.x["3"] = 3
            return self.x

    x = {"1": 1, "2": 2}
    net = Net(x)
    ret = net()
    assert ret == {"1": 10, "2": 2, "3": 3}
    assert net.x == {"1": 10, "2": 2, "3": 3}
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inplace_setitem_with_attribute_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    @jit_class
    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.attr["1"] = 10
            return self.x.attr

    x = {"1": 1, "2": 2}
    obj = AttrClass(x)
    net = Net(obj)
    ret = net()
    assert ret == {"1": 10, "2": 2}
    assert net.x.attr == {"1": 10, "2": 2}
    assert id(x) == id(ret)


@pytest.mark.skip(reason="setitem with abstract any do not convert to pyexecute yet")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inplace_setitem_with_attribute_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """

    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.attr["1"] = 10
            return self.x.attr

    x = {"1": 1, "2": 2}
    obj = AttrClass(x)
    net = Net(obj)
    ret = net()
    assert ret == {"1": 10, "2": 2}
    assert net.x.attr == {"1": 10, "2": 2}
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_getitem_after_setitem():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.dict = {"Name": "b", "Age": 15}

        def construct(self, y):
            self.dict["Age"] = y
            a = (self.dict["Age"] == y)
            return self.dict, a

    net = Net()
    ret1, ret2 = net(Tensor([1]))
    assert ret1 == {"Name": "b", "Age": Tensor([1])}
    assert ret2


global_dict_for_update = {'Name': 'a', 'Age': 7}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_getitem_after_setitem_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    class DcitNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dict = {'Name': 'b', 'Age': 15}

        def construct(self, y, z):
            self.dict['Age'] = y
            if self.dict['Age'] == y:
                global_dict_for_update.update({"Grade": 1})
            z.update({"Grade": "college"})
            return global_dict_for_update, self.dict, z

    y = Tensor([16])
    z = {'Name': 'c', 'Age': 18}
    net = DcitNet()
    ret1, ret2, ret3 = net(y, z)
    assert ret1 == {'Name': 'a', 'Age': 7, 'Grade': 1}
    assert ret2 == {'Name': 'b', 'Age': Tensor([16])}
    assert ret3 == {'Name': 'c', 'Age': 18, 'Grade': 'college'}


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_inplace_setitem_with_dict_getitem():
    """
    Feature: Enable dict inplace operation
    Description: There is dict getitem after inplace operation.
    Expectation: No exception.
    """

    class DictNet(nn.Cell):
        def construct(self, x):
            for key in x:
                x[key] *= 2
            return x[0] + x[1]

    x = {0: Tensor([0]), 1: Tensor([1])}
    ms_out = DictNet()(x)
    ms_grad = ops.grad(DictNet())(x)

    assert ms_out == Tensor([2])
    assert ms_grad == ()
