# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_list_tuple_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test list() and tuple() in fallback runtime
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = list((1, x, np.array([5, 6]), x.asnumpy()))
        b = tuple((1, x, np.array([5, 6]), x.asnumpy()))
        return a, b

    out = foo(Tensor([2, 3]))
    assert isinstance(out[0], list)
    assert isinstance(out[1], tuple)

    assert out[0][0] == 1
    assert (out[0][1] == Tensor([2, 3])).all()
    assert (out[0][2] == np.array([5, 6])).all()
    assert (out[0][3] == np.array([2, 3])).all()

    assert out[1][0] == 1
    assert (out[1][1] == Tensor([2, 3])).all()
    assert (out[1][2] == np.array([5, 6])).all()
    assert (out[1][3] == np.array([2, 3])).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_runtime_list():
    """
    Feature: JIT Fallback
    Description: Test list() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        res = (x.asnumpy(),)
        return list(res)

    x = Tensor(np.arange(0, 6).reshape(2, 3))
    out = foo(x)
    assert (out == x.asnumpy()).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_runtime_tuple():
    """
    Feature: JIT Fallback
    Description: Test tuple() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x, y):
        res = [x.asnumpy()]
        res.append(y)
        return tuple(res)

    x = Tensor(np.arange(0, 6).reshape(2, 3))
    out = foo(x, 3)
    assert (out[0] == x.asnumpy()).all()
    assert out[1] == 3
