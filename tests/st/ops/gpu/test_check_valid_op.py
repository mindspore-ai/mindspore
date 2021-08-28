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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetCheckValid(nn.Cell):
    def __init__(self):
        super(NetCheckValid, self).__init__()
        self.valid = P.CheckValid()

    def construct(self, anchor, image_metas):
        return self.valid(anchor, image_metas)

def check_valid(nptype):
    anchor = np.array([[50, 0, 100, 700], [-2, 2, 8, 100], [10, 20, 300, 2000]], nptype)
    image_metas = np.array([768, 1280, 1], nptype)
    anchor_box = Tensor(anchor)
    image_metas_box = Tensor(image_metas)
    expect = np.array([True, False, False], np.bool)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    boundingbox_decode = NetCheckValid()
    output = boundingbox_decode(anchor_box, image_metas_box)
    assert np.array_equal(output.asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    boundingbox_decode = NetCheckValid()
    output = boundingbox_decode(anchor_box, image_metas_box)
    assert np.array_equal(output.asnumpy(), expect)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_check_valid_float32():
    check_valid(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_check_valid_float16():
    check_valid(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_check_valid_int16():
    check_valid(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_check_valid_uint8():
    anchor = np.array([[5, 0, 10, 70], [2, 2, 8, 10], [1, 2, 30, 200]], np.uint8)
    image_metas = np.array([76, 128, 1], np.uint8)
    anchor_box = Tensor(anchor)
    image_metas_box = Tensor(image_metas)
    expect = np.array([True, True, False], np.bool)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    boundingbox_decode = NetCheckValid()
    output = boundingbox_decode(anchor_box, image_metas_box)
    assert np.array_equal(output.asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    boundingbox_decode = NetCheckValid()
    output = boundingbox_decode(anchor_box, image_metas_box)
    assert np.array_equal(output.asnumpy(), expect)
