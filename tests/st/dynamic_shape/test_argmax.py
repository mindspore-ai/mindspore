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
import pytest
import numpy as np
from mindspore import context
from mindspore import ops

from test_argmin import argmin_argmax_case, argmin_argmax_case_dyn, argmin_argmax_case_vmap
from tests.mark_utils import arg_mark

def argmax_(input_x, axis, output_type):
    return ops.Argmax(axis, output_type)(input_x)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax(mode):
    """
    Feature: Test argmin op.
    Description: Test argmin.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_argmax_case(argmax_, np.argmax)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_vmap(mode):
    """
    Feature: Test argmin op.
    Description: Test argmin vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_argmax_case_vmap(argmax_)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmax_dyn(mode):
    """
    Feature: Test argmin op.
    Description: Test argmin dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_argmax_case_dyn(argmax_, np.argmax)
