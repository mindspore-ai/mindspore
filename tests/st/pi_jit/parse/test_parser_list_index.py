from mindspore.nn import Cell
from mindspore import ops
from mindspore.common import jit
from mindspore.common import Tensor
import sys
import pytest
from tests.mark_utils import arg_mark
from ..share.meta import MetaFactory
from mindspore import Tensor, jit


@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")


@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_parser_list_mul_index_001():
    """
    Feature: Test of parser list
    Description: list 在construct中定义, 2层索引赋值
    Expectation: The outputs of pijit should be equal with mindspore.
    """

    class MsNet(Cell, MetaFactory):
        def __init__(self):
            super().__init__()
            MetaFactory.__init__(self)

        def construct(self):
            list_x = [[1], [2, 3], [4, 5, 6]]
            list_x[2][2] = 9
            return list_x

    class Net(Cell, MetaFactory):
        def __init__(self):
            super().__init__()
            MetaFactory.__init__(self)

        @jit(capture_mode="bytecode")
        def construct(self):
            list_x = [[1], [2, 3], [4, 5, 6]]
            list_x[2][2] = 9
            return list_x

    ms_net = MsNet()
    ms_out = ms_net()

    net = Net()
    out = net()
    assert out == ms_out

@pytest.mark.skip(reason="tmp skip,probabilistic failure")
@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_parser_list_mul_index_009():
    """
    Feature: Test of parser list
    Description: list 从init中传入, 2层索引赋值, 分别赋值为float, str, list, tuple, bool类型
    Expectation: The outputs of pijit should be equal with pynative.
    """

    class Net(Cell, MetaFactory):
        def __init__(self, list_x, input_x):
            super().__init__()
            MetaFactory.__init__(self)
            self.list_x = list_x
            self.input_x = input_x

        @jit(capture_mode="bytecode")
        def construct(self):
            list_x = self.list_x
            list_x[2][1] = self.input_x
            return list_x

    net1 = Net([[1], [2, 2], [3, 3, 3]], 3.4)
    net2 = Net([[1], [2, 2], [3, 3, 3]], "2")
    net3 = Net([[1], [2, 2], [3, 3, 3]], [3, 4])
    net4 = Net([[1], [2, 2], [3, 3, 3]], (3, 4))
    net5 = Net([[1], [2, 2], [3, 3, 3]], True)
    out1 = net1()
    assert round(out1[2][1], 1) == 3.4
    out2 = net2()
    assert out2[2] == [3, "2", 3]
    out3 = net3()
    assert out3[2] == [3, [3, 4], 3]
    out4 = net4()
    assert out4[2] == [3, (3, 4), 3]
    out5 = net5()
    assert out5[2] == [3, True, 3]

@pytest.mark.skip(reason="tmp skip,probabilistic failure")
@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_parser_list_mul_index_012():
    """
    Feature: Test of parser list
    Description: list 从init中传入, 多层索引赋值, 索引取值为负数
    Expectation: The outputs of pijit should be equal with pynative.
    """

    class Net(Cell, MetaFactory):
        def __init__(self, list_x, input_x):
            super().__init__()
            MetaFactory.__init__(self)
            self.list_x = list_x
            self.input_x = input_x

        @jit(capture_mode="bytecode")
        def construct(self):
            list_x = self.list_x
            list_x[-1][1][-2] = self.input_x
            return list_x

    net = Net([[1], [2, 2], [[3, 3], [3, 3]]], 4)
    out = net()
    assert out[2] == [[3, 3], [4, 3]]


@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_parser_list_minus_index_001():
    """
    Feature: Test of parser list
    Description: list从init传入, construct中赋值, 使用负数索引（单层, 多层）, 结合append操作, 多次进行赋值,
    赋值类型包含number, tensor, str, list, tuple, bool类型
    Expectation: The outputs of pijit should be equal with pynative.
    """

    class Net(Cell, MetaFactory):
        def __init__(
            self, list_x, input_1, input_2, input_3, input_4, input_5, input_6
        ):
            super().__init__()
            MetaFactory.__init__(self)
            self.list_x = list_x
            self.input_1 = input_1
            self.input_2 = input_2
            self.input_3 = input_3
            self.input_4 = input_4
            self.input_5 = input_5
            self.input_6 = input_6

        @jit(capture_mode="bytecode")
        def construct(self):
            list_x = self.list_x
            list_x[-3] = self.input_1
            list_x[-2][-1] = self.input_2
            list_x.append(self.input_3)
            list_x[-1][1] = self.input_4
            list_x[-2][0][-1] = self.input_5
            list_x[-2][-1][-1][-3] = self.input_6
            return list_x

    net = Net(
        [1, [2, 2], [[4, 4, 4], [[5, 5, 5]]]],
        Tensor([10]),
        "a",
        [6, 7],
        (8, 9),
        True,
        3,
    )
    out = net()
    assert list(out) == [
        Tensor([10]),
        [2, "a"],
        [[4, 4, True], [[3, 5, 5]]],
        [6, (8, 9)],
    ]


@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_parser_list_minus_index_003():
    """
    Feature: Test of parser list
    Description: list从construct传入, 并赋值, 使用负数索引（单层, 多层）, 结合append操作, 多次进行赋值,
    赋值类型包含number, tensor, str, list, tuple, bool类型
    Expectation: The outputs of pijit should be equal with pynative.
    """

    class Net(Cell, MetaFactory):
        def __init__(self, input_1, input_2, input_3, input_4, input_5, input_6):
            super().__init__()
            MetaFactory.__init__(self)
            self.input_1 = input_1
            self.input_2 = input_2
            self.input_3 = input_3
            self.input_4 = input_4
            self.input_5 = input_5
            self.input_6 = input_6

        @jit(capture_mode="bytecode")
        def construct(self, input_x):
            list_x = input_x
            list_x[-3] = self.input_1
            list_x[-2][-1] = self.input_2
            list_x.append(self.input_3)
            list_x[-1][1] = self.input_4
            list_x[-2][0][-1] = self.input_5
            list_x[-2][-1][-1][-3] = self.input_6
            return list_x

    net = Net(Tensor([10]), "a", [6, 7], (8, 9), True, 3)
    out = net([1, [2, 2], [[4, 4, 4], [[5, 5, 5]]]])
    assert list(out) == [
        Tensor([10]),
        [2, "a"],
        [[4, 4, True], [[3, 5, 5]]],
        [6, (8, 9)],
    ]
