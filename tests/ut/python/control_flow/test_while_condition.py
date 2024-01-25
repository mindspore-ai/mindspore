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
"""test condition of control flow in InferImplSwitch"""
import pytest
from mindspore import jit
from mindspore import Tensor
from mindspore.ops import functional as F


@jit
def defer_resolve(x):
    y1 = x + 1
    y2 = x + 2
    y3 = 1
    y = y1 + y2

    if x > 10:
        y = y + y1 + y3
    else:
        y = y + y2

    y = y + 2
    return y


def test_defer_resolve():
    """
    Feature: defere resolve if true and false branches graphs.
    Description: The if true and false branches graphs will been  resolved at infer phase.
    Expectation: Run successfully
    """
    x = Tensor([10])
    defer_resolve(x)


def test_while_tensor_condition_():
    """
    Feature: Test condition of control flow.
    Description: Tensor condition must be one element or dynamic shape.
    Expectation: No exception.
    """

    @jit
    def foo(cond, x, y):
        return F.switch(cond, x, y)

    x = Tensor([[1, 1], [2, 2]])
    y = Tensor([[3, 3], [4, 4]])
    with pytest.raises(ValueError) as ex:
        foo(x > y, x, y)
    assert "The truth value of an array with more than one element is ambiguous." in str(ex.value)
