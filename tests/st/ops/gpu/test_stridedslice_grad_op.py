# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class StridedSliceGrad(nn.Cell):
    def __init__(self):
        super(StridedSliceGrad, self).__init__()
        self.ssg = G.StridedSliceGrad()
        self.shape = P.Shape()

    @ms_function
    def construct(self, dy, x):
        return self.ssg(dy, self.shape(x), (2, 0, 0), (3, 2, 3), (1, 1, 1))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(np.float32))
    dy = Tensor(np.array([[[5., 1., 5.], [6., 1., 8.]]]).astype(np.float32))
    ssg = StridedSliceGrad()
    output = ssg(dy, x)
    expect = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[5, 1, 5], [6, 1, 8]]]
    assert (output.asnumpy() == expect).all()
