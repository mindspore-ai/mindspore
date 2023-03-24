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
"""test ListComprehension for dynamic sequence in graph mode"""
import pytest
from mindspore.common import mutable
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_dynamic_sequence_list_comp():
    """
    Feature: ListComprehension with dynamic length sequence .
    Description: The dynamic length input is unsupported in graph mode
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable((100, 200, 300, 400), True)
        out = [i + 1 for i in x]
        return out

    with pytest.raises(ValueError) as ex:
        foo()
    assert "For 'ListComprehension' syntax [i for i in x], " in str(ex.value)
    assert "input x can not be dynamic length list/tuple in graph mode" in str(ex.value)
