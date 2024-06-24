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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore.ops.extend import max as max_

from tests.st.ops.test_ops_min import (min_max_case, min_max_case_all_dyn, min_max_case_vmap)
from tests.st.utils.test_utils import compare


def np_max(input_x):
    return np.max(input_x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_dtype', [np.float32])
def test_max(mode, data_dtype):
    """
    Feature: Test max op.
    Description: Test max.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case(max_, np_max, data_dtype=data_dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_nan(mode):
    """
    Feature: Test max op.
    Description: Test max.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case(max_, np_max, has_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_vmap(mode):
    """
    Feature: Test max op.
    Description: Test max vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case_vmap(max_)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('data_dtype', [np.float32])
def test_max_all_dynamic(data_dtype):
    """
    Feature: Test max op.
    Description: Test max with input is dynamic.
    Expectation: the result match with expected result.
    """
    min_max_case_all_dyn(max_, data_dtype=data_dtype)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_dtype', [np.float32])
def test_min_tensor(mode, data_dtype):
    """
    Feature: Test min op.
    Description: Test min.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    x_np = np.random.randn(64, 77).astype(data_dtype)
    x = Tensor(x_np)
    output = x.max()
    expect = np_max(x_np)
    compare(output, expect)
