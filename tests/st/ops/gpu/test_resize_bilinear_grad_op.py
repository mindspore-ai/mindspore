# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations._grad_ops as G
from mindspore import Tensor


class ResizeBilinearGradNet(nn.Cell):
    def __init__(self, align_corners=False, half_pixel_centers=False):
        super(ResizeBilinearGradNet, self).__init__()
        self.rb1 = G.ResizeBilinearGrad(align_corners=align_corners, half_pixel_centers=half_pixel_centers)

    def construct(self, dy, size):
        return self.rb1(dy, size)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_resize_bilinear_grad_align_corners():
    """
    Feature: test ResizeBilinearGrad op
    Description: test ResizeBilinearGrad op
    Expectation: test success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float16)

    x = np.array([[[[1.1, 2.2, 3.2, 2.5],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float16)
    expect = np.array([[[[1., 0., 0., 2.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [3., 0., 0., 4.]]]]).astype(np.float16)
    net = ResizeBilinearGradNet(align_corners=True)
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float32)

    x = np.array([[[[1.1, 2.2, 3.2, 2.5],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float32)
    expect = np.array([[[[1., 0., 0., 2.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [3., 0., 0., 4.]]]]).astype(np.float32)
    net = ResizeBilinearGradNet(align_corners=True)
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_resize_bilinear_grad():
    """
    Feature: test ResizeBilinearGrad op
    Description: test ResizeBilinearGrad op
    Expectation: test success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dy = np.array([[[[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]]]).astype(np.float16)

    x = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(np.float16)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(np.float16)
    net = ResizeBilinearGradNet()
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)

    dy = np.array([[[[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]]]).astype(np.float32)
    x = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(np.float32)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(np.float32)
    net = ResizeBilinearGradNet()
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_resize_bilinear_grad_half_pixel_centers():
    """
    Feature: Test ResizeBilinearGrad on GPU.
    Description:  The half_pixel_centers is True.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float16)

    x = np.array([[[[1.1, 2.2, 3.2, 2.5],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float16)
    expect = np.array([[[[0.25, 0.25, 0.5, 0.5],
                         [0.25, 0.25, 0.5, 0.5],
                         [0.75, 0.75, 1.0, 1.0],
                         [0.75, 0.75, 1.0, 1.0]]]], dtype=np.float16)
    net = ResizeBilinearGradNet(half_pixel_centers=True)
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float32)

    x = np.array([[[[1.1, 2.2, 3.2, 2.5],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float32)
    expect = np.array([[[[0.25, 0.25, 0.5, 0.5],
                         [0.25, 0.25, 0.5, 0.5],
                         [0.75, 0.75, 1.0, 1.0],
                         [0.75, 0.75, 1.0, 1.0]]]], dtype=np.float32)
    net = ResizeBilinearGradNet(half_pixel_centers=True)
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_resize_bilinear_grad_dtype(mode, dtype):
    """
    Feature: Test ResizeBilinearGrad on GPU.
    Description:  Test float16, float32, float64.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=mode, device_target="GPU")
    dy = np.array([[[[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]]]).astype(dtype)

    x = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(dtype)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(dtype)
    net = ResizeBilinearGradNet()
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)
