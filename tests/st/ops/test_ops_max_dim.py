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
import pytest
import numpy as np
from mindspore import context, Tensor
from mindspore.ops.function.array_func import max_ext as max_

from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_ops_min_dim import (
    min_max_dim_case, min_max_dim_case_dyn)

from tests.mark_utils import arg_mark


def np_max(input_x, axis, keepdims):
    value = np.max(input_x, axis)
    index = np.argmax(input_x, axis).astype(np.int32)
    if keepdims:
        value = np.expand_dims(value, axis)
        index = np.expand_dims(index, axis)
    return value, index


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
        level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max(mode):
    """
    Feature: Test max op.
    Description: Test max.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_dim_case(max_, np_max)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.02)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_dyn(mode):
    """
    Feature: Test max op.
    Description: Test max dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_dim_case_dyn(max_, np_max)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.03)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_dyn_rank(mode):
    """
    Feature: Test max op.
    Description: Test max dynamic rank.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_dim_case_dyn(max_, np_max, True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_max_all_dynamic():
    """
    Feature: Test max op.
    Description: Test argmin_with_value with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    t1 = Tensor(np.array([[1, 20], [67, 8]], dtype=np.float32))
    input_case1 = [t1, -1]
    t2 = Tensor(np.array([[[1, 20, 5], [67, 8, 9]], [[130, 24, 15], [16, 64, 32]]], dtype=np.float32))
    input_case2 = [t2, 0]
    TEST_OP(max_, [input_case1, input_case2],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor'])
