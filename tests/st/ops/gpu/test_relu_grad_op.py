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
from mindspore.ops.operations import _grad_ops as G
import mindspore.nn as nn
import numpy as np
import mindspore.context as context


class NetReluGrad(nn.Cell):
    def __init__(self):
        super(NetReluGrad, self).__init__()
        self.rekuGrad = G.ReluGrad()

    def construct(self, x, dy):
        return self.rekuGrad(dy, x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu_grad():
    x = Tensor(np.array([[[[-1, 1, 1],
                           [1, -1, 1],
                           [1, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[1, 0, 1],
                            [0, 1, 0],
                            [1, 1, 1]]]]).astype(np.float32))
    expect = np.array([[[[0, 0, 1, ], [0, 0, 0, ], [1, 1, 0.]]]]).astype(np.float32)
    error = np.ones(shape=[3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu_grad = NetReluGrad()
    output = relu_grad(x, dy)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
