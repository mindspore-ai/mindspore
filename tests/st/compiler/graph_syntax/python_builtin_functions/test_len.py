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
""" test len operation """
import pytest
import numpy as np
from mindspore import jit, context, Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_len_numpy_with_variable():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        x = x + len(np.array([1, 2, 3, 4]))
        return x

    out = foo(Tensor([10]))
    assert out == 14


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_len_list_with_variable():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support list.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        x = x + len([1, 2, 3, 4])
        return x

    out = foo(Tensor([10]))
    assert out == 14


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_len_dict_with_variable():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support dict.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        x = x + len({"1": 1, "2": 2, "3": 3, "4": 4})
        return x

    out = foo(Tensor([10]))
    assert out == 14


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_len_tensor():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tensor.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return len(x), x.__len__()

    x = Tensor(np.array([[1, 2, 3], [0, 0, 0]]))
    out = foo(x)
    assert out[0] == out[1] == 2
