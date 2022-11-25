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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Net(nn.Cell):
    def __init__(self, coordinate_transformation_mode, mode):
        super(Net, self).__init__()
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.mode = mode

    def construct(self, x, scales=None, sizes=None):
        return ops.interpolate(x, None, scales, sizes, self.coordinate_transformation_mode, self.mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_interpolate_normal(mode):
    """
    Feature: interpolate
    Description: Verify the result of interpolate
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net("align_corners", "bicubic")
    x = ops.arange(4, dtype=ms.float32).reshape((1, 1, 2, 2))
    out1 = net(x, sizes=(4, 6))
    expect_out1 = np.array([[[[0.0000, 0.1762, 0.3884, 0.6116, 0.8238, 1.0000],
                              [0.6289, 0.8051, 1.0174, 1.2405, 1.4527, 1.6289],
                              [1.3711, 1.5473, 1.7595, 1.9826, 2.1949, 2.3711],
                              [2.0000, 2.1762, 2.3884, 2.6116, 2.8238, 3.0000]]]])
    assert np.allclose(out1.asnumpy(), expect_out1, rtol=1e-4)
    out2 = net(x, scales=(1.0, 1.0, 3.0, 3.0))
    expect_out2 = np.array([[[[0.0000, 0.1762, 0.3884, 0.6116, 0.8238, 1.0000],
                              [0.3524, 0.5286, 0.7408, 0.9640, 1.1762, 1.3524],
                              [0.7769, 0.9531, 1.1653, 1.3884, 1.6007, 1.7769],
                              [1.2231, 1.3993, 1.6116, 1.8347, 2.0469, 2.2231],
                              [1.6476, 1.8238, 2.0360, 2.2592, 2.4714, 2.6476],
                              [2.0000, 2.1762, 2.3884, 2.6116, 2.8238, 3.0000]]]])
    assert np.allclose(out2.asnumpy(), expect_out2, rtol=1e-4)
