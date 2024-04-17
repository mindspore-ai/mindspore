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
"""run dynamic shape test"""
import pytest
from mindspore import Tensor, jit, context

@jit(mode="PIJit", jit_config={"enable_dynamic_shape": True, "limit_graph_count": 1})
def dynamic_shape_test(a, b):
    return a + b

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_shape_case():
    """
    Feature: Method DynamicShape Testing
    Description: Test dyanmicshape function to check whether it works.
    Expectation: The result of the case should dump the dynamic shape ir at last.
                 'enable_dynamic_shape' flag is used to enable dynamic shape when calling 3 times for different shape.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1])
    b = Tensor([2])
    expect = Tensor([3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1])
    b = Tensor([2, 2])
    expect = Tensor([3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    expect = Tensor([3, 3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
