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


@test_utils.run_with_cell
def geqrf_forward_func(x):
    return ops.geqrf(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_geqrf_forward(mode):
    """
    Feature: Ops.
    Description: test op geqrf.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[-2.0, -1.0], [1.0, 2.0]]).astype(np.float32))
    expect_out1 = np.array([[2.236068, 1.7888544],
                            [-0.236068, 1.3416407]])
    expect_out2 = np.array([1.8944271, 0.])
    out = geqrf_forward_func(x)
    assert np.allclose(out[0].asnumpy(), expect_out1)
    assert np.allclose(out[1].asnumpy(), expect_out2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_geqrf_vmap(mode):
    """
    Feature: test vmap function.
    Description: test geqrf op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.array([[[[-2.0, -1.0], [1.0, 2.0]]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(
        geqrf_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    expect_out1 = np.array([[[[-2.00000000e+00]],
                             [[1.00000000e+00]]],
                            [[[-1.00000000e+00]],
                             [[2.00000000e+00]]]])
    expect_out2 = np.array([[[0.00000000e+00],
                             [0.00000000e+00]],
                            [[0.00000000e+00],
                             [0.00000000e+00]]])
    assert np.allclose(out[0].asnumpy(), expect_out1)
    assert np.allclose(out[1].asnumpy(), expect_out2)
