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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

from mindspore.ops.functional import vmap
from mindspore.ops import functional as F
from mindspore.common.api import jit


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class CdistNet(nn.Cell):
    def __init__(self, p):
        super(CdistNet, self).__init__()
        self.cdist = P.Cdist(p)

    def construct(self, x1, x2):
        return self.cdist(x1, x2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdist_p_2_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 2.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistNet(2.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array(
        [[[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(np.float32)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdist_p_0_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 0.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistNet(0.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[2.0, 2.0], [2.0, 2.0]]]).astype(np.float32)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdist_p_1_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 1.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistNet(1.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[4.0, 4.0], [2.0, 2.0]]]).astype(np.float32)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdist_p_8_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 8.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistNet(8.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array(
        [[[2.1810155, 2.1810155], [1.0905077, 1.0905077]]]).astype(np.float32)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdist_p_inf_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = inf.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistNet(float('inf'))
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[2., 2.], [1., 1.]]]).astype(np.float32)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdist_p_2_float32_func():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 2.0.
    Expectation: the output[0] is same as numpy
    """
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    expect = np.array(
        [[[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(np.float32)
    output = F.cdist(x1, x2, 2.0)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    print(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap():
    """
    Feature: cdist vmap.
    Description: test the rightness of cdist vmap feature.
    Expectation: Success.
    """

    def cal_cdist(x, y):
        return P.Cdist(2.0)(x, y)

    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                          [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                          [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                          [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))

    expect = np.array([[[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(np.float32)

    vmap_cdist = vmap(cal_cdist, in_axes=(0), out_axes=0)
    output = vmap_cdist(x1, x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    @jit
    def manually_batched(xs, ys):
        output = []
        for i in range(xs.shape[0]):
            output.append(cal_cdist(xs[i], ys[i]))
        return F.stack(output)

    expect_m = manually_batched(x1, x2)
    np.testing.assert_allclose(output.asnumpy(), expect_m.asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap2():
    """
    Feature: cdist vmap.
    Description: test the rightness of cdist vmap feature.
    Expectation: Success.
    """

    def cal_cdist(x, y):
        return P.Cdist(2.0)(x, y)

    x1 = Tensor(np.array([[[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                           [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                           [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                           [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]],
                          [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                           [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                           [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                           [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))

    expect = np.array([[[[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]]],
                       [[[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                        [[2.828427, 2.828427], [1.4142135, 1.4142135]]]]).astype(np.float32)

    vmap_cdist = vmap(vmap(cal_cdist, in_axes=(0), out_axes=0), in_axes=(0, None), out_axes=0)
    output = vmap_cdist(x1, x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
