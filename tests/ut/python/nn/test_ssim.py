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
test ssim
"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.api import _executor
from mindspore import Tensor


class SSIMNet(nn.Cell):
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIMNet, self).__init__()
        self.net = nn.SSIM(max_val, filter_size, filter_sigma, k1, k2)

    def construct(self, img1, img2):
        return self.net(img1, img2)


def test_compile():
    net = SSIMNet()
    img1 = Tensor(np.random.random((8, 3, 16, 16)))
    img2 = Tensor(np.random.random((8, 3, 16, 16)))
    _executor.compile(net, img1, img2)

def test_compile_grayscale():
    max_val = 255
    net = SSIMNet(max_val = max_val)
    img1 = Tensor(np.random.randint(0, 256, (8, 1, 16, 16), np.uint8))
    img2 = Tensor(np.random.randint(0, 256, (8, 1, 16, 16), np.uint8))
    _executor.compile(net, img1, img2)

def test_ssim_max_val_negative():
    max_val = -1
    with pytest.raises(ValueError):
        net = SSIMNet(max_val)

def test_ssim_max_val_bool():
    max_val = True
    with pytest.raises(TypeError):
        net = SSIMNet(max_val)

def test_ssim_max_val_zero():
    max_val = 0
    with pytest.raises(ValueError):
        net = SSIMNet(max_val)

def test_ssim_filter_size_float():
    with pytest.raises(TypeError):
        net = SSIMNet(filter_size=1.1)

def test_ssim_filter_size_zero():
    with pytest.raises(ValueError):
        net = SSIMNet(filter_size=0)

def test_ssim_filter_sigma_zero():
    with pytest.raises(ValueError):
        net = SSIMNet(filter_sigma=0.0)

def test_ssim_filter_sigma_negative():
    with pytest.raises(ValueError):
        net = SSIMNet(filter_sigma=-0.1)

def test_ssim_k1_k2_wrong_value():
    with pytest.raises(ValueError):
        net = SSIMNet(k1=1.1)
    with pytest.raises(ValueError):
        net = SSIMNet(k1=1.0)
    with pytest.raises(ValueError):
        net = SSIMNet(k1=0.0)
    with pytest.raises(ValueError):
        net = SSIMNet(k1=-1.0)

    with pytest.raises(ValueError):
        net = SSIMNet(k2=1.1)
    with pytest.raises(ValueError):
        net = SSIMNet(k2=1.0)
    with pytest.raises(ValueError):
        net = SSIMNet(k2=0.0)
    with pytest.raises(ValueError):
        net = SSIMNet(k2=-1.0)

def test_ssim_different_shape():
    shape_1 = (8, 3, 16, 16)
    shape_2 = (8, 3, 8, 8)
    img1 = Tensor(np.random.random(shape_1))
    img2 = Tensor(np.random.random(shape_2))
    net = SSIMNet()
    with pytest.raises(ValueError):
        _executor.compile(net, img1, img2)

def test_ssim_different_dtype():
    dtype_1 = mstype.float32
    dtype_2 = mstype.float16
    img1 = Tensor(np.random.random((8, 3, 16, 16)), dtype=dtype_1)
    img2 = Tensor(np.random.random((8, 3, 16, 16)), dtype=dtype_2)
    net = SSIMNet()
    with pytest.raises(TypeError):
        _executor.compile(net, img1, img2)

def test_ssim_invalid_5d_input():
    shape_1 = (8, 3, 16, 16)
    shape_2 = (8, 3, 8, 8)
    invalid_shape = (8, 3, 16, 16, 1)
    img1 = Tensor(np.random.random(shape_1))
    invalid_img1 = Tensor(np.random.random(invalid_shape))
    img2 = Tensor(np.random.random(shape_2))
    invalid_img2 = Tensor(np.random.random(invalid_shape))

    net = SSIMNet()
    with pytest.raises(ValueError):
        _executor.compile(net, invalid_img1, img2)
    with pytest.raises(ValueError):
        _executor.compile(net, img1, invalid_img2)
    with pytest.raises(ValueError):
        _executor.compile(net, invalid_img1, invalid_img2)
