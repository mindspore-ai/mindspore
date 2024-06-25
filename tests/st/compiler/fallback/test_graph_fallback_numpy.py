# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import pytest
import numpy as np
from mindspore import jit, context, Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


# Not support <class 'complex'> yet.
@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_np_array_advanced_index_complex():
    """
    Feature: JIT Fallback
    Description: Test numpy with array advanced index in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_array_advanced_index_2():
        x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        y = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
        z = np.array([1, 2 + 6j, 5, 3.5 + 5j])
        a = Tensor(x[x > 5])
        b = Tensor(y[~np.isnan(y)])
        c = Tensor(z[np.iscomplex(z)])
        return a, b, c
    a, b, c = np_array_advanced_index_2()
    assert np.all(a.asnumpy() == np.array([6, 7, 8, 9, 10, 11]))
    assert np.all(b.asnumpy() == np.array([1., 2., 3., 4., 5.]))
    assert np.all(c.asnumpy() == np.array([2. + 6.j, 3.5 + 5.j]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_np_rollaxis():
    """
    Feature: JIT Fallback
    Description: Test numpy.rollaxis() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_rollaxis():
        x = np.arange(8).reshape(2, 2, 2)
        tensor_x = Tensor(x)
        y = np.rollaxis(x, 2, 0)
        tensor_y = Tensor(y)
        return tensor_x[1, 1, 0], tensor_y[1, 1, 0]
    x, y = np_rollaxis()
    assert x == 6 and y == 5


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_np_swapaxes():
    """
    Feature: JIT Fallback
    Description: Test numpy.swapaxes() method in graph mode.
    Expectation: No exception.
    """
    @jit
    def np_swapaxes():
        x = np.arange(8).reshape(2, 2, 2)
        tensor_x = Tensor(x)
        y = np.swapaxes(x, 2, 0)
        tensor_y = Tensor(y)
        return tensor_x[1, 1, 0], tensor_y[1, 1, 0]
    x, y = np_swapaxes()
    assert x == 6 and y == 3
