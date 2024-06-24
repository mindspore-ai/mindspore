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
from mindspore import context, Tensor
from mindspore.ops.extend import max as max_

from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.ops.test_ops_min_dim import (argmin_with_value_argmax_case, argmin_with_value_argmax_case_dyn,
                                           argmin_with_value_argmax_case_vmap)

def np_argmax_with_value(input_x, axis, keepdims):
    value = np.max(input_x, axis)
    index = np.argmax(input_x, axis).astype(np.int32)
    if keepdims:
        value = np.expand_dims(value, axis)
        index = np.expand_dims(index, axis)
    return value, index


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case(max_, np_argmax_with_value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_vmap(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case_vmap(max_)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_dyn(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case_dyn(max_, np_argmax_with_value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_with_value_dyn_rank(mode):
    """
    Feature: Test argmax_with_value op.
    Description: Test argmax_with_value dynamic rank.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_with_value_argmax_case_dyn(max_, np_argmax_with_value, True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_argmax_with_value_all_dynamic():
    """
    Feature: Test argmax_with_value op.
    Description: Test argmin_with_value with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    t1 = Tensor(np.array([[1, 20], [67, 8]], dtype=np.float32))
    input_case1 = [t1, -1]
    t2 = Tensor(np.array([[[1, 20, 5], [67, 8, 9]], [[130, 24, 15], [16, 64, 32]]], dtype=np.float32))
    input_case2 = [t2, 0]
    TEST_OP(max_, [input_case1, input_case2], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
