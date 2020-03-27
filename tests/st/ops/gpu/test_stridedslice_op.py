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

import pytest
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class StridedSlice(nn.Cell):
    def __init__(self):
        super(StridedSlice, self).__init__()
        self.stridedslice = P.StridedSlice()

    def construct(self, x):
        return self.stridedslice(x, (2, 0, 0), (3, 2, 3), (1, 1, 1))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(np.int32))
    stridedslice = StridedSlice()
    output = stridedslice(x)
    expect = [[[5., 5., 5.],
               [6., 7., 8.]]]
    assert (output.asnumpy() == expect).all()
