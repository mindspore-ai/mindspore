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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations.image_ops import ResizeBicubic
import mindspore.common.dtype as mstype
import pytest


class NetResizeBicubic(nn.Cell):

    def __init__(self):
        super(NetResizeBicubic, self).__init__()
        align_corners = False
        half_pixel_centers = False
        self.resize_bicubic = ResizeBicubic(align_corners, half_pixel_centers)

    def construct(self, images, size):
        return self.resize_bicubic(images, size)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bicubic_graph():
    """
    Feature: test operations running in graph mode
    Description: test in gragh mode
    Expectation: success or throw pytest error
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    types = [np.float16, np.float32, np.float64]
    for type_i in types:
        img = np.array([1, 2, 3, 4])
        img = img.reshape([1, 1, 2, 2])
        images = Tensor(img.astype(type_i))
        size = Tensor([1, 4], mstype.int32)
        net = NetResizeBicubic()
        output = net(images, size)
        expect_type = output.asnumpy().dtype
        expect = np.array([1, 1.5, 2, 2.09375])
        expect = expect.reshape([1, 1, 1, 4])

        expect = expect.astype(np.float32)
        assert expect_type == 'float32'
        assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bicubic_pynative():
    """
    Feature: test operations in result and output type
    Description: test in pynative mode on GPU
    Expectation: success or throw pytest error
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    types_2 = [np.float16, np.float32, np.float64]
    for type_i in types_2:
        img = np.array([1, 2, 3, 4])
        img = img.reshape([1, 1, 2, 2])
        images = Tensor(img.astype(type_i))
        size = Tensor([1, 4], mstype.int32)
        net = NetResizeBicubic()
        output = net(images, size)
        expect_type_2 = output.asnumpy().dtype
        expect = np.array([1, 1.5, 2, 2.09375])
        expect = expect.reshape([1, 1, 1, 4])

        expect = expect.astype(np.float32)
        assert expect_type_2 == 'float32'
        assert (output.asnumpy() == expect).all()
