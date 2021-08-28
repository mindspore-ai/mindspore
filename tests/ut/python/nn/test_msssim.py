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
test msssim
"""
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor

_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

class MSSSIMNet(nn.Cell):
    def __init__(self, max_val=1.0, power_factors=_MSSSIM_WEIGHTS, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(MSSSIMNet, self).__init__()
        self.net = nn.MSSSIM(max_val, power_factors, filter_size, filter_sigma, k1, k2)

    def construct(self, img1, img2):
        return self.net(img1, img2)


def test_compile():
    factors = (0.033, 0.033, 0.033)
    net = MSSSIMNet(power_factors=factors)
    img1 = Tensor(np.random.random((8, 3, 128, 128)))
    img2 = Tensor(np.random.random((8, 3, 128, 128)))
    _cell_graph_executor.compile(net, img1, img2)


def test_compile_grayscale():
    max_val = 255
    factors = (0.033, 0.033, 0.033)
    net = MSSSIMNet(max_val=max_val, power_factors=factors)
    img1 = Tensor(np.random.randint(0, 256, (8, 3, 128, 128), np.uint8))
    img2 = Tensor(np.random.randint(0, 256, (8, 3, 128, 128), np.uint8))
    _cell_graph_executor.compile(net, img1, img2)


def test_msssim_max_val_negative():
    max_val = -1
    with pytest.raises(ValueError):
        _ = MSSSIMNet(max_val)


def test_msssim_max_val_bool():
    max_val = True
    with pytest.raises(TypeError):
        _ = MSSSIMNet(max_val)


def test_msssim_max_val_zero():
    max_val = 0
    with pytest.raises(ValueError):
        _ = MSSSIMNet(max_val)


def test_msssim_power_factors_set():
    with pytest.raises(TypeError):
        _ = MSSSIMNet(power_factors={0.033, 0.033, 0.033})


def test_msssim_filter_size_float():
    with pytest.raises(TypeError):
        _ = MSSSIMNet(filter_size=1.1)


def test_msssim_filter_size_zero():
    with pytest.raises(ValueError):
        _ = MSSSIMNet(filter_size=0)


def test_msssim_filter_sigma_zero():
    with pytest.raises(ValueError):
        _ = MSSSIMNet(filter_sigma=0.0)


def test_msssim_filter_sigma_negative():
    with pytest.raises(ValueError):
        _ = MSSSIMNet(filter_sigma=-0.1)


def test_msssim_different_shape():
    shape_1 = (8, 3, 128, 128)
    shape_2 = (8, 3, 256, 256)
    factors = (0.033, 0.033, 0.033)
    img1 = Tensor(np.random.random(shape_1))
    img2 = Tensor(np.random.random(shape_2))
    net = MSSSIMNet(power_factors=factors)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(net, img1, img2)


def test_msssim_different_dtype():
    dtype_1 = mstype.float32
    dtype_2 = mstype.float16
    factors = (0.033, 0.033, 0.033)
    img1 = Tensor(np.random.random((8, 3, 128, 128)), dtype=dtype_1)
    img2 = Tensor(np.random.random((8, 3, 128, 128)), dtype=dtype_2)
    net = MSSSIMNet(power_factors=factors)
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, img1, img2)


def test_msssim_invalid_5d_input():
    shape_1 = (8, 3, 128, 128)
    shape_2 = (8, 3, 256, 256)
    invalid_shape = (8, 3, 128, 128, 1)
    factors = (0.033, 0.033, 0.033)
    img1 = Tensor(np.random.random(shape_1))
    invalid_img1 = Tensor(np.random.random(invalid_shape))
    img2 = Tensor(np.random.random(shape_2))
    invalid_img2 = Tensor(np.random.random(invalid_shape))

    net = MSSSIMNet(power_factors=factors)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(net, invalid_img1, img2)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(net, img1, invalid_img2)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(net, invalid_img1, invalid_img2)
