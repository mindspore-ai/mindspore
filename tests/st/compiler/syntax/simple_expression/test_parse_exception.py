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

import pytest

import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.ops as ops
import mindspore
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class TestNoReturn(nn.Cell):
    def __init__(self):
        super(TestNoReturn, self).__init__()
        self.m = 1

    def construct(self, x, y):
        and_v = x * y
        and_v += 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_no_return():
    """
    Feature: simple expression
    Description: test has no return
    Expectation: No exception
    """
    net = TestNoReturn()
    x = Tensor(np.ones([2, 2], np.float))
    y = Tensor(np.zeros([2, 2], np.float))
    ret = net(x, y)
    assert not ret


class TestCompare(nn.Cell):
    def construct(self, x, y, z):
        return x > y > z


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_compare():
    """
    Feature: simple expression
    Description: test compare
    Expectation: No exception
    """
    net = TestCompare()
    x = Tensor([3])
    y = Tensor([2])
    z = Tensor([1])
    ret = net(x, y, z)
    assert ret


class TestMemberChange(nn.Cell):
    def __init__(self):
        super(TestMemberChange, self).__init__()
        self.t = Tensor(np.zeros([2, 2], np.float))

    def construct(self, x, y):
        self.t = x
        return x > y


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_member_changer():
    """
    Feature: simple expression
    Description: test class member
    Expectation: No exception
    """
    net = TestMemberChange()
    x = Tensor(np.ones([2, 2], np.float))
    y = Tensor(np.zeros([2, 2], np.float))
    ret = net(x, y)
    assert ret.all()


class TestUnsupportSTMT(nn.Cell):
    def __init__(self):
        super(TestUnsupportSTMT, self).__init__()
        self.m = 1

    def construct(self, x, y):
        try:
            val = x + y
        finally:
            val = x
        return val


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_unsupport_stmt():
    """
    Feature: simple expression
    Description: test try
    Expectation: No exception
    """
    with pytest.raises(RuntimeError) as err:
        net = TestUnsupportSTMT()
        x = Tensor(np.ones([2, 2], np.float))
        y = Tensor(np.zeros([2, 2], np.float))
        ret = net(x, y)
        print(ret)
    assert "Unsupported statement 'Try'." in str(err)


class TestUnsupportNum(nn.Cell):
    def __init__(self):
        super(TestUnsupportNum, self).__init__()
        self.m = 1

    def construct(self, x, y):
        a = x + 3.14j
        return a


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_unsupport_num():
    """
    Feature: simple expression
    Description: test complex num
    Expectation: No exception
    """
    with pytest.raises(TypeError) as err:
        net = TestUnsupportNum()
        x = Tensor(np.ones([2, 2], np.float))
        y = Tensor(np.zeros([2, 2], np.float))
        ret = net(x, y)
        print(ret)
    assert "Only support 'Number' type of 'int` and 'float'" in str(err)


class TestAssignAdd(nn.Cell):
    def __init__(self):
        super(TestAssignAdd, self).__init__()
        self.m = 1

    def construct(self, x, y):
        x.id_ += y
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_assign_add():
    """
    Feature: simple expression
    Description: test assignadd
    Expectation: No exception
    """
    with pytest.raises(AttributeError) as err:
        net = TestAssignAdd()
        ret = net([3, 1], 2)
        print(ret)
    assert "'List' object has no attribute 'id_'" in str(err)


class TestParseListComp(nn.Cell):
    def __init__(self):
        super(TestParseListComp, self).__init__()
        self.m = 1

    def construct(self, x, y):
        ret = [m + y for l in x for m in l]
        return ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parse_list_comp():
    """
    Feature: simple expression
    Description: test ListComp
    Expectation: No exception
    """
    with pytest.raises(TypeError) as err:
        net = TestParseListComp()
        ret = net([[1, 2], [3, 4]], 2)
        print(ret)
    assert "The 'generators' supports 1 'comprehension' in ListComp/GeneratorExp, but got 2 comprehensions." in str(err)


class TestAssign(nn.Cell):
    def __init__(self):
        super(TestAssign, self).__init__()
        self.m = 1

    def construct(self, x, y):
        x.id_ = y
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_assign():
    """
    Feature: simple expression
    Description: test assign
    Expectation: No exception
    """
    with pytest.raises(AttributeError) as err:
        net = TestAssign()
        ret = net([3, 1], 2)
        print(ret)
    assert "'list' object has no attribute 'id_'" in str(err)


class TestAssignList(nn.Cell):
    def __init__(self):
        super(TestAssignList, self).__init__()
        self.m = 1

    def construct(self, x, y):
        [m, n] = [x, y]
        return m, n


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_assign_list():
    """
    Feature: simple expression
    Description: test assign
    Expectation: No exception
    """
    net = TestAssignList()
    ret = net([3, 1], 2)
    assert ret == ([3, 1], 2)


class TestParaDef(nn.Cell):
    def __init__(self):
        super(TestParaDef, self).__init__()
        self.m = 1

    def construct(self, x=1, y=1):
        ret = x + y
        return ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_para_def():
    """
    Feature: simple expression
    Description: test def graph
    Expectation: No exception
    """
    net = TestParaDef()
    ret = net(1, 2)
    assert ret == 3


class TestParameterNameNone(nn.Cell):
    def __init__(self):
        super(TestParameterNameNone, self).__init__()
        self.matmul = ops.MatMul()
        self.weight = Parameter(Tensor(np.ones((1, 2)), mindspore.float32), name=None, requires_grad=True)

    def construct(self, x):
        out = self.matmul(self.weight, x)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parameter_name_none():
    """
    Feature: simple expression
    Description: test parameter name
    Expectation: No exception
    """
    net = TestParameterNameNone()
    x = Tensor(np.ones((2, 1)), mindspore.float32)
    out = net(x)
    assert out.asnumpy() == [[2.0]]


class TestBranchReturn(nn.Cell):
    def __init__(self):
        super(TestBranchReturn, self).__init__()
        self.m = 1

    def construct(self, x):
        if x > 0:
            return x + 1
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_branch_return():
    """
    Feature: simple expression
    Description: test return in control flow
    Expectation: No exception
    """
    net = TestBranchReturn()
    out = net(1)
    assert out == 2


class TestSliceNotInt(nn.Cell):
    def __init__(self):
        super(TestSliceNotInt, self).__init__()
        self.m = 1

    def construct(self, x):
        s = "ABCDEFGHIJKL"
        sl = slice(x, 4.5)
        return s[sl]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_slice_not_int():
    """
    Feature: simple expression
    Description: test slice
    Expectation: No exception
    """
    with pytest.raises(RuntimeError) as err:
        net = TestSliceNotInt()
        print(net(1))
    assert "Attribute 'stop' of slice(1, 4.5, None) should be int or Tensor with Int type but got 4.5" in str(err)


class TestSliceNotIntDefInInit(nn.Cell):
    def __init__(self):
        super(TestSliceNotIntDefInInit, self).__init__()
        self.sl = slice(1, 4.5)

    def construct(self, x):
        s = "ABCDEFGHIJKL"
        return s[self.sl]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_slice_not_int_def_in_init():
    """
    Feature: simple expression
    Description: test slice
    Expectation: No exception
    """
    with pytest.raises(RuntimeError) as err:
        net = TestSliceNotIntDefInInit()
        print(net(1))
    assert "Attribute 'stop' of slice(1, 4.5, None) should be int or Tensor with Int type but got 4.5" in str(err)


class MatMulCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.m = 1

    def construct(self, x):
        return x


class TestCellPipelineStage(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.block = nn.CellList()
        cell = MatMulCell()
        cell.pipeline_stage = -1
        self.block.append(cell)
        cell = MatMulCell()
        cell.pipeline_stage = -1
        self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_cell_pipeline_state():
    """
    Feature: simple expression
    Description: test cell pipeline state
    Expectation: No exception
    """
    with pytest.raises(ValueError) as err:
        strategy1 = Tensor((4, 1), mindspore.int64)
        strategy2 = Tensor((2, 1), mindspore.int64)
        net = TestCellPipelineStage(strategy1, strategy2)
        print(net(1))
    assert "For 'Cell', the property 'pipeline_stage' can not be less than 0, but got -1" in str(err)


class TestArgsKwArgs(nn.Cell):
    def __init__(self):
        super(TestArgsKwArgs, self).__init__()
        self.m = 1

    def construct(self, *args, **kwargs):
        x = 0
        for v in args:
            x += v
        for _, v in kwargs.items():
            x += v
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_args_kwargs():
    """
    Feature: simple expression
    Description: test args and kwargs
    Expectation: No exception
    """
    net = TestArgsKwArgs()
    out = net(1, 2, 3, 4, k1=5, k2=6)
    assert out == 21


class TestArgs(nn.Cell):
    def __init__(self):
        super(TestArgs, self).__init__()
        self.m = 1

    def construct(self, x, *args):
        for v in args:
            x += v
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_args():
    """
    Feature: simple expression
    Description: test args
    Expectation: No exception
    """
    net = TestArgs()
    out = net(1, 2, 3, 4)
    assert out == 10


class TestNoDef(nn.Cell):
    def __init__(self):
        super().__init__()
        self.m = 1

    def construct(self, x):
        x += self.y
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_no_def():
    """
    Feature: simple expression
    Description: test def graph
    Expectation: No exception
    """
    with pytest.raises(AttributeError) as err:
        net = TestNoDef()
        print(net(1))
    assert "External object has no attribute y" in str(err)
