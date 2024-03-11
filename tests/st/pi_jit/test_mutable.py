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
"""run mutable test"""
import pytest
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor, jit, context

@jit(mode="PIJit")
def is_mutable():
    output = mutable((Tensor([1]), Tensor([2])), True)
    return F.is_sequence_value_unknown(output), F.is_sequence_shape_unknown(output)

@jit(mode="PIJit")
def not_mutable():
    output = mutable((Tensor([1]), Tensor([2])), False)
    return F.is_sequence_value_unknown(output), F.is_sequence_shape_unknown(output)


@pytest.mark.skip(reason="pynative mode and graph mode, results is not equal")
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('fun', [is_mutable])
def test_mutable_case1(fun):
    """
    Feature: Method Mutable Testing
    Description: Test mutable function to set the attribute for dynamic length of object.
    Expectation: The result of the case1 should check whether both value and shape are unknown.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    unknown_value, unknown_shape = fun()
    assert unknown_value
    assert unknown_shape


@pytest.mark.skip(reason="pynative mode and graph mode, results is not equal")
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('fun', [not_mutable])
def test_mutable_case2(fun):
    """
    Feature: Method Mutable Testing
    Description: Test mutable function to set the attribute for dynamic length of object.
    Expectation: The result of the case2 should check whether value is unknown and shape is defined.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    unknown_value, unknown_shape = fun()
    assert unknown_value
    assert not unknown_shape
