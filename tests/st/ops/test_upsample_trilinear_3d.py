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

import numpy as np
import pytest
import os
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations.nn_ops import UpsampleTrilinear3D


class UpsampleTrilinear3DNet(nn.Cell):

    def __init__(self, align_corners=False):
        super(UpsampleTrilinear3DNet, self).__init__()
        self.upsample = UpsampleTrilinear3D(align_corners=align_corners)

    def construct(self, x, output_size, scales):
        return self.upsample(x, output_size, scales)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_upsample_trilinear_3d_dynamic_shape(mode):
    """
    Feature: Test UpsampleTrilinear3D op in ascend with scales is None.
    Description: Test UpsampleTrilinear3D op in ascend with scales is None.
    Expectation: Expect correct shape result.
    """
    if mode == ms.GRAPH_MODE:
        os.environ['GRAPH_OP_RUN'] = '1'
    ms.set_context(mode=mode, device_target='Ascend')
    net = UpsampleTrilinear3DNet()
    x = Tensor(np.random.randn(2, 5, 60, 30, 128), dtype=ms.float16)
    output_size = (4, 64, 32)
    scales = None
    output = net(x, output_size, scales)
    expect_shape = (2, 5, 4, 64, 32)
    assert expect_shape == output.asnumpy().shape
