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
import os
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_map_lambda_with_fv():
    """
    Feature: Support map for lambda with FV.
    Description: Support map for lambda with FV.
    Expectation: No exception.
    """
    os.environ['MS_DEV_PRE_LIFT'] = '1'

    @ms.jit()
    def map_lambda_with_fv(x, y, z):
        number_add = lambda x, y: x + y + z
        return map(number_add, (x,), (y,))

    res = map_lambda_with_fv(1, 5, 9)
    del os.environ['MS_DEV_PRE_LIFT']
    assert res == (15,)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
