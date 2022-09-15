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

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


class StridedSlice(nn.Cell):
    def __init__(self, begin=(2, 0, 0), end=(3, 2, 3), strides=(1, 1, 1)):
        super().__init__()
        self.stridedslice = P.StridedSlice()
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x):
        return self.stridedslice(x, self.begin, self.end, self.strides)


class StridedSliceDynamicRank(nn.Cell):
    def __init__(self, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
        super().__init__()
        self.stridedslice = P.StridedSlice(
            begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
        self.reduce_sum = P.ReduceSum()

    def construct(self, x, begin, end, strides, axis):
        x = self.reduce_sum(x, axis)
        return self.stridedslice(x, begin, end, strides)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stridedslice_dynamic_shape():
    """
    Feature: Dynamic shape.
    Description: Test StridedSlice dynamic shape.
    Expectation: Success.
    """
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(np.float32))
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    net = StridedSlice()
    net.set_inputs(x_dyn)
    output = net(x)
    expect = [[[5., 5., 5.],
               [6., 7., 8.]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stridedslice_dynamic_rank():
    """
    Feature: Dynamic rank.
    Description: Test StridedSlice dynamic rank.
    Expectation: Success.
    """
    begin_mask = (0, 1, 2)
    end_mask = (4, 5, 6)
    strides_mask = (1, 1, 2)
    x = Tensor(np.random.randn(5, 6, 7, 8, 9).astype(np.float32))
    begin_ms = Tensor(begin_mask, mstype.int64)
    end_ms = Tensor(end_mask, mstype.int64)
    strides_ms = Tensor(strides_mask, mstype.int64)
    axis_ms = Tensor(np.array([0, 0, 1, 1]))

    net = StridedSliceDynamicRank()
    out = net(x, begin_ms, end_ms, strides_ms, axis_ms)

    axis_dyn = Tensor(shape=(None,), dtype=axis_ms.dtype)
    net_dyn = StridedSliceDynamicRank()
    net_dyn.set_inputs(x, begin_ms, end_ms, strides_ms, axis_dyn)
    out_dyn = net_dyn(x, begin_ms, end_ms, strides_ms, axis_ms)
    assert(out.asnumpy() == out_dyn.asnumpy()).all()
