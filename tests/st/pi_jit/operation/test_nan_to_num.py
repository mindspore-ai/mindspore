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
"""Test ops.nan_to_num()"""

import numpy as np

import mindspore as ms
from mindspore import Tensor, context, ops, jit

from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config

jit_cfg = {'compile_with_try': False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_nan_to_num():
    """
    Feature: ops.nan_to_num()
    Description: Specify nan, inf, -inf value.
    Expectation: No exception, no graph-break.
    """

    def fn(x: Tensor):
        return ops.nan_to_num(x, nan=1.0, posinf=2.0, neginf=3.0)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 5.0]), ms.float32)
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_nan_to_num_with_default_params():
    """
    Feature: ops.nan_to_num()
    Description: Do not specify nan, inf, -inf value, use default value.
    Expectation: No exception, no graph-break.
    """

    def fn(x: Tensor):
        return ops.nan_to_num(x)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 5.0]), ms.float32)
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_NanToNum():
    """
    Feature: ops.NanToNum()
    Description: Specify nan, inf, -inf value.
    Expectation: No exception, no graph-break.
    """

    def fn(x: Tensor):
        return ops.NanToNum(nan=1.0, posinf=2.0, neginf=3.0)(x)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]), ms.float32)
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_NanToNum_with_default_params():
    """
    Feature: ops.NanToNum()
    Description: Do not specify nan, inf, -inf value, use default value.
    Expectation: No exception, no graph-break.
    """

    def fn(x: Tensor):
        return ops.NanToNum()(x)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]), ms.float32)
    o1 = fn(x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)
