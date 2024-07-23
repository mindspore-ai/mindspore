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

import numpy as np
import pytest
from tests.st.utils import test_utils

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def __init__(self, dtype):
        super(Net, self).__init__()
        self.Cast = ops.Cast()
        self.dtype = dtype

    def construct(self, x):
        return self.Cast(x, self.dtype)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_cast_forward(mode):
    """
    Feature: Ops.
    Description: test op cast.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    output = Net(ms.float64)(input_x)
    expect = np.array([1.1, 2.5, -1.5]).astype(np.float64)
    assert np.allclose(output.asnumpy(), expect)
    assert output.asnumpy().dtype == 'float64'


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_cast_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op cast.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    grads = ops.grad(Net(ms.float64), (0,))(input_x)
    expect = np.array([1, 1, 1]).astype(np.float32)
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_cast_vmap(mode):
    """
    Feature: test vmap function.
    Description: test cast op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0)
    np_x = np.array([[[1.1, 0.9], [2.2, 1.8]]]).astype(np.float32)
    x = ms.Tensor(np_x)
    expect = np_x = np.array([[[1.1, 0.9], [2.2, 1.8]]]).astype(np.float64)
    nest_vmap = ops.vmap(Net(ms.float64), in_axes=in_axes, out_axes=0)
    vmap_out = nest_vmap(x)
    assert np.allclose(vmap_out.asnumpy(), expect)
    assert vmap_out.asnumpy().dtype == 'float64'
