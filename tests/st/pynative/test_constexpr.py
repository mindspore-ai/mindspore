# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest
from mindspore.ops.primitive import constexpr
from mindspore.common.api import jit


def _temp_func():
    return 0


@constexpr(check=False)
def _is_need_compile(func):
    # No matter what the value of mode is, in ms_function scenario, this function always returns true.
    return func is None


@jit
def run_in_ms_function():
    return _is_need_compile(_temp_func)


@constexpr
def run_in_pyhon(func):
    # No matter what the value of mode is, in ms_function scenario, this function always returns true.
    return func is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_constexpr():
    """
    Feature: test const expr
    Description: test const expr in python native and graph
    Expectation: success
    """
    assert not run_in_pyhon(_temp_func)
    assert run_in_ms_function()
