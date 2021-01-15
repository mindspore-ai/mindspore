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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner

class MatMul_d(nn.Cell):
    def __init__(self):
        super(MatMul_d, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.matmul = P.MatMul()

    def construct(self, x, y):
        x = self.test_dynamic(x)
        y = self.test_dynamic(y)
        return self.matmul(x, y)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_MatMul_dynamic():

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = MatMul_d()

    x1 = np.arange(2).reshape(1, 2).astype(np.float32)
    y1 = np.arange(4).reshape(2, 2).astype(np.float32)
    output1 = net(Tensor(x1), Tensor(y1))
    expect1 = np.matmul(x1, y1)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect1)

    x2 = np.arange(102).reshape(34, 3).astype(np.float32)
    y2 = np.arange(18).reshape(3, 6).astype(np.float32)
    output2 = net(Tensor(x2), Tensor(y2))
    expect2 = np.matmul(x2, y2)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect2)
