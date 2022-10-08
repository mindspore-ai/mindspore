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


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=3, strides=1, pads=1)

    def construct(self, x):
        out = self.pool(x)
        return out


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=3, strides=1, pads=1, return_indices=True)

    def construct(self, x):
        out = self.pool(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_maxpool3d_normal(mode):
    """
    Feature: MaxPool3d
    Description: Verify the result of MaxPool3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_array = np.arange(1 * 2 * 4 * 4 * 5).reshape((1, 2, 4, 4, 5))

    net1 = Net()
    net2 = Net2()
    x = ms.Tensor(np_array, ms.float32)
    output1 = net1(x)
    output2 = net2(x)
    output3 = np.array([[[[[26.0, 27.0, 28.0, 29.0, 29.0], [31.0, 32.0, 33.0, 34.0, 34.0],
                           [36.0, 37.0, 38.0, 39.0, 39.0], [36.0, 37.0, 38.0, 39.0, 39.0]],
                          [[46.0, 47.0, 48.0, 49.0, 49.0], [51.0, 52.0, 53.0, 54.0, 54.0],
                           [56.0, 57.0, 58.0, 59.0, 59.0], [56.0, 57.0, 58.0, 59.0, 59.0]],
                          [[66.0, 67.0, 68.0, 69.0, 69.0], [71.0, 72.0, 73.0, 74.0, 74.0],
                           [76.0, 77.0, 78.0, 79.0, 79.0], [76.0, 77.0, 78.0, 79.0, 79.0]],
                          [[66.0, 67.0, 68.0, 69.0, 69.0], [71.0, 72.0, 73.0, 74.0, 74.0],
                           [76.0, 77.0, 78.0, 79.0, 79.0], [76.0, 77.0, 78.0, 79.0, 79.0]]],
                         [[[106.0, 107.0, 108.0, 109.0, 109.0], [111.0, 112.0, 113.0, 114.0, 114.0],
                           [116.0, 117.0, 118.0, 119.0, 119.0], [116.0, 117.0, 118.0, 119.0, 119.0]],
                          [[126.0, 127.0, 128.0, 129.0, 129.0], [131.0, 132.0, 133.0, 134.0, 134.0],
                           [136.0, 137.0, 138.0, 139.0, 139.0], [136.0, 137.0, 138.0, 139.0, 139.0]],
                          [[146.0, 147.0, 148.0, 149.0, 149.0], [151.0, 152.0, 153.0, 154.0, 154.0],
                           [156.0, 157.0, 158.0, 159.0, 159.0], [156.0, 157.0, 158.0, 159.0, 159.0]],
                          [[146.0, 147.0, 148.0, 149.0, 149.0], [151.0, 152.0, 153.0, 154.0, 154.0],
                           [156.0, 157.0, 158.0, 159.0, 159.0], [156.0, 157.0, 158.0, 159.0, 159.0]]]]])
    output4 = np.array([[[[[26, 27, 28, 29, 29], [31, 32, 33, 34, 34], [36, 37, 38, 39, 39], [36, 37, 38, 39, 39]],
                          [[46, 47, 48, 49, 49], [51, 52, 53, 54, 54], [56, 57, 58, 59, 59], [56, 57, 58, 59, 59]],
                          [[66, 67, 68, 69, 69], [71, 72, 73, 74, 74], [76, 77, 78, 79, 79], [76, 77, 78, 79, 79]],
                          [[66, 67, 68, 69, 69], [71, 72, 73, 74, 74], [76, 77, 78, 79, 79], [76, 77, 78, 79, 79]]],
                         [[[26, 27, 28, 29, 29], [31, 32, 33, 34, 34], [36, 37, 38, 39, 39], [36, 37, 38, 39, 39]],
                          [[46, 47, 48, 49, 49], [51, 52, 53, 54, 54], [56, 57, 58, 59, 59], [56, 57, 58, 59, 59]],
                          [[66, 67, 68, 69, 69], [71, 72, 73, 74, 74], [76, 77, 78, 79, 79], [76, 77, 78, 79, 79]],
                          [[66, 67, 68, 69, 69], [71, 72, 73, 74, 74], [76, 77, 78, 79, 79], [76, 77, 78, 79, 79]]]]])
    assert output1.shape == (1, 2, 4, 4, 5)
    assert len(output2) == 2
    assert output2[1].shape == (1, 2, 4, 4, 5)
    assert np.allclose(output1.asnumpy(), output3)
    assert np.allclose(output2[0].asnumpy(), output1.asnumpy())
    assert np.allclose(output2[1].asnumpy(), output4)
