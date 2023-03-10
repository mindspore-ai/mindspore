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
"""test mutable with dynamic length"""
import pytest
from mindspore.common import mutable
from mindspore import Tensor
from mindspore import jit
from mindspore import context


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_generate_mutable_sequence_with_dynamic_length_with_jit():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output1 = mutable([1, 2, 3, 4], True)
        output2 = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
        output3 = mutable([(1, 2, 3), (2, 3, 4), (3, 4, 5)], True)
        return output1, output2, output3
    ret = foo()
    assert len(ret) == 3
    assert ret[0] == [1, 2, 3, 4]
    assert ret[1] == [Tensor([1]), Tensor([2]), Tensor([3])]
    assert ret[2] == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
