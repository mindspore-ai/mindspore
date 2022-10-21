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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class StridedSliceGrad(nn.Cell):
    def __init__(self):
        super(StridedSliceGrad, self).__init__()
        self.ssg = G.StridedSliceGrad()
        self.shape = P.Shape()

    @jit
    def construct(self, dy, x):
        return self.ssg(dy, self.shape(x), (2, 0, 0), (3, 2, 3), (1, 1, 1))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(np.float32))
    dy = Tensor(np.array([[[5., 1., 5.], [6., 1., 8.]]]).astype(np.float32))
    ssg = StridedSliceGrad()
    output = ssg(dy, x)
    expect = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[5, 1, 5], [6, 1, 8]]]
    assert (output.asnumpy() == expect).all()


class StridedSliceGrad2(nn.Cell):
    def __init__(self):
        super(StridedSliceGrad2, self).__init__()
        self.ssg = G.StridedSliceGrad()
        self.shape = P.Shape()

    @jit
    def construct(self, dy, x):
        return self.ssg(dy, self.shape(x), (0, 0, 0), (1, 4, 2), (1, 1, 1))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice2():
    x = Tensor(np.arange(2 * 4 * 2).reshape(2, 4, 2), mstype.float32)
    dy = Tensor(np.arange(4 * 2).reshape(4, 2), mstype.float32)
    ssg = StridedSliceGrad2()
    output = ssg(dy, x)
    expect = [[[0., 1.], [2., 3.], [4., 5.], [6., 7.]], [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice_vmap():
    """
    Feature: Test stridedslicegrad CPU vmap.
    Description: The inputs are two tensors, x and dy.
    Expectation: The output matches to the np benchmark.
    """
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 7, 8]]]).astype(np.float32))
    dy = P.Stack()([Tensor(np.array([[[5., 1., 5.], [6., 1., 8.]]]).astype(np.float32)) for _ in range(8)])
    ssg_vmap = vmap(StridedSliceGrad(), in_axes=(0, None))
    output = ssg_vmap(dy, x)
    expect = P.Stack()([Tensor(np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                                         [[5, 1, 5], [6, 1, 8]]])) for _ in range(8)]).asnumpy()
    assert (output.asnumpy() == expect).all()
