# Copyright 2024 Huawei Technologies Co., Ltd
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
"""run os.env test"""
import pytest
import os
import numpy as np
import mindspore as ms
from mindspore._c_expression import get_code_extra
from mindspore import Tensor, jit, context


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_os_env_mapping_get():
    """
    Feature: collections.abc.Mapping.get
    Description: get os.env key by collections.abc.Mapping.get
    Expectation: 0 break count
    """
    def func():
        device_id = os.environ.get("DEVICE_ID")
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_os_env_mapping_get_with_set():
    """
    Feature: collections.abc.Mapping.get
    Description: get os.env key by collections.abc.Mapping.get
    Expectation: 0 break count
    """
    def func():
        device_id = os.environ.get("DEVICE_ID")
    os.environ["DEVICE_ID"] = "3"
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("a", [ms.Tensor(np.random.randn(2, 2).astype(np.float32))])
@pytest.mark.parametrize("b", [ms.Tensor(np.random.randn(2, 2).astype(np.float32))])
def test_os_env_mapping_get_with_tensor(a, b):
    """
    Feature: collections.abc.Mapping.get
    Description: get os.env key by collections.abc.Mapping.get
    Expectation: 0 break count
    """
    def func(a, b):
        if os.environ.get("DEVICE_ID"):
            return a * b
        return a + b
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(fn=func, mode="PIJit", jit_config={"compile_by_trace": False})(a, b) # One-stage will fix it later
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0
