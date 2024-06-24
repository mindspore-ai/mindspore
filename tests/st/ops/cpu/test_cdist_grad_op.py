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

import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class CdistGradTEST(nn.Cell):
    def __init__(self, p):
        super(CdistGradTEST, self).__init__()
        self.cdist_grad = G.CdistGrad(p)

    def construct(self, grad, x1, x2, dist):
        return self.cdist_grad(grad, x1, x2, dist)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_CdistGradP0_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 0.0.
    Expectation: the output[0] is same as numpy
    """
    cdist_grad = CdistGradTEST(3.)
    grad = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    dist = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist_grad(grad, x1, x2, dist)
    expect = np.array(
        [[[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]]]).astype(np.float32)
    print(output)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_vmap():
    """
    Feature: cdist vmap.
    Description: test the rightness of cdist vmap feature.
    Expectation: Success.
    """

    def cal_cdist_grad(grad, x1, x2, dist):
        return G.CdistGrad(3.0)(grad, x1, x2, dist)

    grad = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    dist = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    expect = np.array(
        [[[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]],
         [[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]]]).astype(np.float32)

    vmap_ceil = vmap(cal_cdist_grad, in_axes=(0), out_axes=0)
    output = vmap_ceil(grad, x1, x2, dist)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap2():
    """
    Feature: cdist vmap.
    Description: test the rightness of cdist vmap feature.
    Expectation: Success.
    """

    def cal_cdist_grad(grad, x1, x2, dist):
        return G.CdistGrad(3.0)(grad, x1, x2, dist)

    grad = Tensor(np.array([[[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]],
                            [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]]).astype(np.float32))
    x1 = Tensor(np.array([[[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]],
                          [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]]).astype(np.float32))
    x2 = Tensor(np.array([[[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]],
                          [[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]]).astype(np.float32))
    dist = Tensor(np.array([[[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]],
                            [[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]]).astype(np.float32))
    expect = np.array(
        [[[[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]],
          [[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]]],
         [[[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]],
          [[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]]]]).astype(np.float32)

    vmap_ceil = vmap(vmap(cal_cdist_grad, in_axes=(
        0), out_axes=0), in_axes=(0), out_axes=0)
    output = vmap_ceil(grad, x1, x2, dist)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
