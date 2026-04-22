import mindspore.nn as nn
import numpy as np
import sys
import pytest
from tests.mark_utils import arg_mark
from ..parse.parser_factory import ParserFactory


@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")


@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")
def test_parser_tensor_fancy_index_bool_list_in_2d_001():
    """
    Feature: Test of pijit parser
    Description: index bool list [[False,True,True],[True,False]], shape[3,2] outshape(1,)
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            x = x[[False,True,True],[True,False]]
            x = self.relu(x)
            return x
    
    net_ms = Net()
    net_pi = Net()
    input_np = np.random.randn(3,2).astype(np.float32)
    fact = ParserFactory(net_ms,net_pi,input_np)
    fact.forward_cmp()
    fact.backward_cmp()

