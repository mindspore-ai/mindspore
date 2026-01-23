# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Tests for view operations.
"""
import pytest
from tests.mark_utils import arg_mark
from tests.st.ops.share.view_dtype_ops import ViewDtypeOpsFactory
from tests.st.ops.share._op_info.op_database import get_op_info


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("mode", ['pynative'])
@pytest.mark.parametrize("op_info", ['Tensor.view(dtype)'])
def test_view_dtype_reference_forward(mode, op_info):
    """
    Feature: Tensor.view(dtype) operations
    Description: Compare forward results.
    Expectation: MindSpore matches the benchmark for outputs.
    """
    fact = ViewDtypeOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("mode", ['pynative'])
@pytest.mark.parametrize("op_info", ['Tensor.view(dtype)'])
def test_view_dtype_reference_backward(mode, op_info):
    """
    Feature: Tensor.view(dtype) operations
    Description: Compare gradients.
    Expectation: MindSpore matches the benchmark for gradients.
    """
    fact = ViewDtypeOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference(grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("mode", ['pynative'])
@pytest.mark.parametrize("op_info", ['Tensor.view(dtype)'])
def test_view_dtype_error(mode, op_info):
    """
    Feature: Tensor.view(dtype) operations
    Description: Test view_dtype error cases.
    Expectation: Run success without error.
    """
    fact = ViewDtypeOpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_error()
