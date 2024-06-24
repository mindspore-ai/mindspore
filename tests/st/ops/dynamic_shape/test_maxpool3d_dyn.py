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

"""test Maxpool3d dynamic shape"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore import nn


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.pool = ops.MaxPool3D(kernel_size=2, strides=1, pad_mode="valid")

    def construct(self, x):
        out = self.pool(x)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_maxpool3d_valid(mode):
    """
    Feature: MaxPool3d
    Description: test dynamic shape of MaxPool3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)), ms.float32)
    expected_shape = (1, 2, 1, 1, 2)

    net = Net()
    x_dyn = ms.Tensor(shape=[None]*len(x.shape), dtype=x.dtype)
    net.set_inputs(x_dyn)
    output = net(x)
    assert output.shape == expected_shape
