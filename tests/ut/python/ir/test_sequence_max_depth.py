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
"""test mutable"""
import pytest
from mindspore import jit


def test_list_max_depth():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the first scalar input.
    Expectation: Get the correct gradient.
    """
    @jit
    def foo():
        x = [[1], [2, 3], [4, [[5, [6, [7, [8]]]]]]]
        x[2][1][0][1][1][1][0] = 9
        return x

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "List, tuple and dict nesting is not allowed more than" in str(ex.value)


def test_list_max_depth_2():
    """
    Feature: Set Constants mutable.
    Description: Get gradient with respect to the first scalar input.
    Expectation: Get the correct gradient.
    """
    @jit
    def foo():
        x = ([1], [2, 3], [4, [[5, [6, [7, [8]]]]]])
        return x

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "List, tuple and dict nesting is not allowed more than" in str(ex.value)
