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
"""test the feature of boost parse"""
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import context, jit, Tensor
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_equal_np_int8_1():
    """
    Feature: Support constant folding for parse.
    Description: Folding the equal judgement of int8 number.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.num = np.int8(2)

        def construct(self):
            if self.num == 2:
                return 1
            raise ValueError("The self.num should be 2.")

    net = Net()
    out = net()
    assert out == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_equal_np_int8_2():
    """
    Feature: Support constant folding for parse.
    Description: Folding the equal judgement of int8 number.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.num = np.int8(1)

        def construct(self):
            if self.num == 2:
                return 1
            raise ValueError("The self.num should be 2.")

    with pytest.raises(ValueError) as raise_info:
        net = Net()
        net()
    assert "The self.num should be 2." in str(raise_info)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_constant_condition_comment():
    """
    Feature: Support constant folding for parse.
    Description: Folding the equal judgement according to the comment.
    Expectation: Get the correct result.
    """

    @jit
    def f1(x):
        if x == 2:  # @jit.cond: True
            return 1
        return 0

    @jit
    def f2(x):
        if x == 2:  # @jit.cond: False
            return 1
        return 0

    x = Tensor(2)
    assert f1(x) == 1
    assert f2(x) == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_if_with_raise():
    """
    Feature: Support constant folding for parse.
    Description: Nested if and there is a raise statement in the inner if while the outer if can not be folded.
    Expectation: Get the correct result.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = Tensor(2)
            self.y = 1

        def construct(self):
            z = 2
            if self.x > 1:
                if self.y == 1:  # pylint: disable=no-else-raise
                    raise RuntimeError('Some error')
                else:
                    z = z + 1
            return z

    with pytest.raises(RuntimeError) as raise_info:
        net = Net()
        net()
    assert "Some error" in str(raise_info)
