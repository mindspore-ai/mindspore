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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common.api import jit
from mindspore import Tensor
from mindspore.ops.operations.image_ops import AdjustHue


class AdHue(nn.Cell):
    def __init__(self):
        super().__init__()
        self.adjusthue = AdjustHue()

    @jit
    def construct(self, input_images, hue_delta):
        return self.adjusthue(input_images, hue_delta)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_adjusthue_float32():
    """
    Feature: None
    Description: basic test float32
    Expectation: just test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    inputimages = np.array([[[100, 100, 100], [2, 2, 2]]]).astype(np.float32)
    huedelta = 0.2
    net = AdHue()
    out = net(Tensor(inputimages, dtype=ms.float32), Tensor(huedelta, dtype=ms.float32))
    expect_out = np.array([[[100, 100, 100], [2, 2, 2]]]).astype(np.float32)
    np.allclose(out.asnumpy(), expect_out, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_adjusthue_float64():
    """
    Feature: None
    Description: basic test float64
    Expectation: just test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    inputimages = np.array([[[100, 100, 100], [2, 2, 2]]]).astype(np.float64)
    huedelta = 0.2
    net = AdHue()
    out = net(Tensor(inputimages, dtype=ms.float64), Tensor(huedelta, dtype=ms.float32))
    expect_out = np.array([[[100, 100, 100], [2, 2, 2]]]).astype(np.float64)
    np.allclose(out.asnumpy(), expect_out, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_adjusthue_float16():
    """
    Feature: None
    Description: basic test float64
    Expectation: just test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    inputimages = np.array([[[100, 100, 100], [2, 2, 2]]]).astype(np.float16)
    huedelta = 0.2
    net = AdHue()
    out = net(Tensor(inputimages, dtype=ms.float16), Tensor(huedelta, dtype=ms.float32))
    expect_out = np.array([[[100, 100, 100], [2, 2, 2]]]).astype(np.float16)
    np.allclose(out.asnumpy(), expect_out, 0.0001, 0.0001)
