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

"""test augassign mod."""

import pytest
from mindspore import Tensor, jit
from mindspore import ops
from tests.mark_utils import arg_mark


@jit(backend="ms_backend")
def foo1(x, y):
    x %= y
    return x


@jit(backend="GE")
def foo2(x, y):
    x %= y
    return x


def foo3(x, y):
    x = ops.abs(x)
    x %= y
    return x


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("y", [3, Tensor(3)])
def test_augassign_mod(y):
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace.
    Expectation: Run success.
    """
    x1 = Tensor(10)
    graph_ret = foo1(x1, y)

    x2 = Tensor(10)
    ge_graph_ret = foo2(x2, y)

    x3 = Tensor(10)
    pynative_ret = foo3(x3, y)
    assert graph_ret == pynative_ret
    assert ge_graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=3072.01)
@pytest.mark.parametrize("y", [3, Tensor(3)])
@pytest.mark.parametrize("grad", [ops.GradOperation(), ops.GradOperation(get_all=True)])
def test_augassign_mod_grad(y, grad):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    x1 = Tensor(10)
    graph_ret = grad(foo1)(x1, y)

    x2 = Tensor(10)
    ge_graph_ret = grad(foo2)(x2, y)

    x3 = Tensor(10)
    pynative_ret = grad(foo3)(x3, y)
    assert graph_ret == pynative_ret
    assert ge_graph_ret == pynative_ret
