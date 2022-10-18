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
import mindspore.context as context
from mindspore.common.api import jit
from mindspore import Tensor
from mindspore.ops.operations.image_ops import AdjustSaturation


class AdSaturation(nn.Cell):
    def __init__(self):
        super().__init__()
        self.adjustsaturation = AdjustSaturation()

    @jit
    def construct(self, input_images, saturation_scale):
        return self.adjustsaturation(input_images, saturation_scale)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adjustsaturation_float32():
    """
    Feature: None
    Description: basic test float32
    Expectation: just test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    inputimages = np.array([[[1, 1, 1], [2, 2, 2]]]).astype(np.float32)
    saturation_scale = 0.5
    net = AdSaturation()
    out = net(Tensor(inputimages, dtype=ms.float32), Tensor(saturation_scale, dtype=ms.float32))
    expect_out = np.array([[[1, 1, 1], [2, 2, 2]]]).astype(np.float32)
    np.allclose(out.asnumpy(), expect_out, 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adjustsaturation_float64():
    """
    Feature: None
    Description: basic test float64
    Expectation: just test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    inputimages = np.array([[[1, 1, 1], [2, 2, 2]]]).astype(np.float64)
    saturation_scale = 0.5
    net = AdSaturation()
    out = net(Tensor(inputimages, dtype=ms.float64), Tensor(saturation_scale, dtype=ms.float32))
    expect_out = np.array([[[1, 1, 1], [2, 2, 2]]]).astype(np.float64)
    np.allclose(out.asnumpy(), expect_out, 0.00001, 0.00001)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adjustsaturation_float16():
    """
    Feature: None
    Description: basic test float16
    Expectation: just test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    inputimages = np.array([[[1, 1, 1], [2, 2, 2]]]).astype(np.float16)
    saturation_scale = 0.5
    net = AdSaturation()
    out = net(Tensor(inputimages, dtype=ms.float16), Tensor(saturation_scale, dtype=ms.float32))
    expect_out = np.array([[[1, 1, 1], [2, 2, 2]]]).astype(np.float16)
    np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)
