# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest
import mindspore as ms
from mindspore.ops.function.math_func import F
import mindspore.context as context
from mindspore import Tensor


context.set_context(device_target='CPU')


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_index_add(mode, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for InplaceIndexAdd
    Expectation: the result match to numpy
    """
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(dtype))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    updates = Tensor(np.array([[1, 2], [7, 8]]).astype(dtype))
    x = F.inplace_index_add(x, indices, updates, axis=0)
    expect = [[2, 4], [10, 12], [5, 6]]
    assert np.allclose(x.asnumpy(), expect, rtol=1e-6)
