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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
from mindspore import context
from mindspore.ops import maximum

from tests.st.ops.test_ops_minimum import (minimum_maximum_case, minimum_maximum_case_vmap,
                                           minimum_maximum_case_all_dyn)

def np_maximum(input_x, input_y):
    return np.maximum(input_x, input_y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    minimum_maximum_case(maximum, np_maximum, is_minimum=False)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum_vmap(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    minimum_maximum_case_vmap(maximum)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maximum_all_dynamic():
    """
    Feature: Test minimum op.
    Description: Test minimum with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    minimum_maximum_case_all_dyn(maximum)
