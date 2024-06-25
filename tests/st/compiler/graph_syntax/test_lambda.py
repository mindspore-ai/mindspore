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
""" test map for lambda with fv. """
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore._extends.parse import compile_config
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="Removed Pre-Lift action.")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_map_lambda_with_fv():
    """
    Feature: Support map for lambda with FV.
    Description: Support map for lambda with FV.
    Expectation: No exception.
    """
    saved_config = compile_config.PRE_LIFT
    compile_config.PRE_LIFT = 1

    @ms.jit()
    def map_lambda_with_fv(x, y, z):
        number_add = lambda x, y: x + y + z
        return map(number_add, (x,), (y,))

    res = map_lambda_with_fv(1, 5, 9)
    compile_config.PRE_LIFT = saved_config
    assert res == (15,)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_lambda_location():
    """
    Feature: Added error message for lambda expressions.
    Description: Added error message for lambda expressions.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, combine_fn=lambda x: x + 1):
            super(Net, self).__init__()
            self.combine_fn = combine_fn

        def construct(self, x):
            out = self.combine_fn(x)
            return out

    with pytest.raises(ValueError) as info:
        net = Net()
        out = net(4)
        assert out == 5
    assert "An error occurred while parsing the positional information of the lambda expression." in str(info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_lambda_location_2():
    """
    Feature: Added error message for lambda expressions.
    Description: Added error message for lambda expressions.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, y):
            super(Net, self).__init__()
            self.func = nn.ReLU() if y < 1 else lambda x: x + 1

        def construct(self, x):
            out = self.func(x)
            return out

    with pytest.raises(TypeError) as info:
        net = Net(2)
        out = net(4)
        print("out:", out)
        assert out == 5
    assert "Parse Lambda Function Fail." in str(info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_lambda_location_3():
    """
    Feature: Added error message for lambda expressions.
    Description: Added error message for lambda expressions.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, y):
            super(Net, self).__init__()
            lambda_func = lambda x: x + 1
            self.func = nn.ReLU() if y < 1 else lambda_func

        def construct(self, x):
            out = self.func(x)
            return out

    net = Net(2)
    out = net(4)
    assert out == 5
