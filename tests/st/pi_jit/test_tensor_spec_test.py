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
"""run tensor specialized test"""
import pytest
from mindspore import Tensor, jit, context
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark

zero = Tensor([0], mstype.int32)
one = Tensor([1], mstype.int32)
five = Tensor([5], mstype.int32)

@jit(mode="PIJit")
def fr(x):
    y = zero
    if x < 0:
        y = one
    elif x < 3:
        y = x * fr(x - 1)
    elif x < 5:
        y = x * fr(x - 2)
    else:
        y = fr(x - 4)
    z = y + 1
    return z


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('fun', [fr])
def test_tensor_spec_case(fun):
    """
    Feature: Method Tensor Specializing Testing
    Description: Test tensor specializing function when pruning case.
    Expectation: The result of the case should test whether the tensor specilizing can work.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    ret = fr(five)
    expect = Tensor([3], mstype.int32)
    assert ret == expect
