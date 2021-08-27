# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
test CentralCrop
"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor


class CentralCropNet(nn.Cell):
    def __init__(self, central_fraction):
        super(CentralCropNet, self).__init__()
        self.net = nn.CentralCrop(central_fraction)

    def construct(self, image):
        return self.net(image)


def test_compile_3d_central_crop():
    central_fraction = 0.2
    net = CentralCropNet(central_fraction)
    image = Tensor(np.random.random((3, 16, 16)), mstype.float32)
    _cell_graph_executor.compile(net, image)


def test_compile_4d_central_crop():
    central_fraction = 0.5
    net = CentralCropNet(central_fraction)
    image = Tensor(np.random.random((8, 3, 16, 16)), mstype.float32)
    _cell_graph_executor.compile(net, image)


def test_central_fraction_bool():
    central_fraction = True
    with pytest.raises(TypeError):
        _ = CentralCropNet(central_fraction)


def test_central_crop_central_fraction_negative():
    central_fraction = -1.0
    with pytest.raises(ValueError):
        _ = CentralCropNet(central_fraction)


def test_central_fraction_zero():
    central_fraction = 0.0
    with pytest.raises(ValueError):
        _ = CentralCropNet(central_fraction)


def test_central_crop_invalid_5d_input():
    invalid_shape = (8, 3, 16, 16, 1)
    invalid_image = Tensor(np.random.random(invalid_shape))

    net = CentralCropNet(central_fraction=0.5)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(net, invalid_image)
