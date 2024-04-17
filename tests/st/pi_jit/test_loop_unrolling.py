import pytest
from mindspore import Tensor, jit, ops
from mindspore import numpy as np
import mindspore.nn as nn
from mindspore import context
from .share.utils import match_array


class ListTest():
    list = list()

    def __init__(self, *args):
        a = Tensor(ops.fill(np.float32, (2, 2), 1))
        b = Tensor(ops.fill(np.float32, (2, 2), 2))
        c = Tensor(ops.fill(np.float32, (2, 2), 3))
        self.list.append(a)
        self.list.append(b)
        self.list.append(c)

    def test(self):
        elem = None
        for elem in self.list:
            elem = elem + elem
        return elem

    @jit(mode="PIJit")
    def test_pi_jit(self):
        elem = None
        for elem in self.list:
            elem = elem + elem
        return elem

    def test_sideeffect(self, x):
        self.list.append(x)
        elem = None
        for elem in self.list:
            elem = elem + elem
        self.list.pop()
        return elem

    @jit(mode="PIJit")
    def test_sideeffect_pi_jit(self, x):
        self.list.append(x)
        elem = None
        for elem in self.list:
            elem = elem + elem
        self.list.pop()
        return elem


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_list():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    test = ListTest()
    context.set_context(mode=context.PYNATIVE_MODE)
    res = test.test()
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = test.test_pi_jit()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


class CellListTest(nn.Cell):
    def __init__(self, *args):
        super().__init__()
        conv = nn.Conv2d(3, 64, 3)
        bn = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        self.cell_list = nn.CellList([conv])
        self.cell_list.append(bn)
        self.cell_list.append(relu)

    def test(self, x):
        for cell in self.cell_list:
            x = cell(x)
        return x

    @jit(mode="PIJit")
    def test_pi_jit(self, x):
        for cell in self.cell_list:
            x = cell(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('input_x', [Tensor(ops.fill(np.float32, (1, 3, 64, 32), 8))])
def test_celllist(input_x):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    test = CellListTest()
    context.set_context(mode=context.PYNATIVE_MODE)
    res = test.test(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = test.test_pi_jit(input_x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip(reason="the pointer[GetDevicePtr] is null")
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('input_x', [Tensor(ops.fill(np.float32, (2, 2), 4))])
def test_sideeffect(input_x):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    test = ListTest()
    context.set_context(mode=context.PYNATIVE_MODE)
    res = test.test_sideeffect(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = test.test_sideeffect_pi_jit(input_x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
