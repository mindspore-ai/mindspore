# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

context.set_context(device_target="Ascend")

class Net(nn.Cell):
    def construct(self, theta, out_size):
        return ops.affine_grid(theta, out_size, False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_affine_grid(mode):
    """
    Feature: test affine_grid forward.
    Description: test GRAPH and PYNATIVE mode for ascend.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode)
    net = Net()
    value = Tensor([[[[-0.78333336, -0.06666666], [-0.25, -0.4], [0.28333336, -0.733333335]],
                     [[-0.28333336, 0.733333335], [0.25, 0.4], [0.78333336, 0.06666666]]]], mindspore.float32)
    theta = Tensor([[[0.8, 0.5, 0], [-0.5, 0.8, 0]]], mindspore.float32)
    out_size = (1, 3, 2, 3)
    output = net(theta, out_size)
    np.testing.assert_allclose(output.asnumpy(), value.asnumpy(), 0.00001)
