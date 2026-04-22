from mindspore.nn import Cell
import sys
import pytest
from tests.mark_utils import arg_mark
from ..parse.parser_factory import ParserFactory
from mindspore.common import jit
from mindspore._c_expression import get_code_extra


@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")


@pytest.mark.skip(reason='fix it later')
@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_pijit_bytecode_build_inner_class():
    """
    Feature: Test of parser inner class
    Description: Test of parser inner class
    Expectation: process success.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()

        @jit(capture_mode="bytecode")
        def construct(self, x):
            class MyClass:
                p = 2
            def inner():
                a = MyClass()
                return a.p * x
            out = x * inner()
            return out
    net_pi = Net()
    got = net_pi(3)
    assert got == 18
    jcr = get_code_extra(Net.construct.__wrapped__)
    assert jcr["break_count_"] == 1
