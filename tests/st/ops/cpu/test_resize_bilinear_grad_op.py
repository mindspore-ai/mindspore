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

import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class ResizeBilinearGradAlignCornerT(nn.Cell):
    def __init__(self):
        super(ResizeBilinearGradAlignCornerT, self).__init__()
        self.op = G.ResizeBilinearGrad(align_corners=True)

    def construct(self, dy, size):
        return self.op(dy, size)


class ResizeBilinearGradAlignCornerF(nn.Cell):
    def __init__(self):
        super(ResizeBilinearGradAlignCornerF, self).__init__()
        self.op = G.ResizeBilinearGrad(align_corners=False)

    def construct(self, dy, size):
        return self.op(dy, size)


class NetResizeBilinearFunc(nn.Cell):
    def construct(self, inputs, size, align_corner=False, half_pixel_centers=False):
        if align_corner and not half_pixel_centers:
            return ops.ResizeBilinearV2(align_corners=True, half_pixel_centers=False)(inputs, size)
        return ops.ResizeBilinearV2(align_corners=False, half_pixel_centers=True)(inputs, size)


def test_resize_bilinear_grad_align_corner():
    """
    Feature: Test ResizeBilinearGrad on CPU.
    Description:  Test align corner true.
    Expectation: Assert that results are consistent with expect.
    """
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float32)

    orign_image = np.array(
        [[[[1.1, 2.2, 3.2, 2.5], [3.3, 4.4, 5.7, 8.1], [3.3, 4.4, 5.7, 8.1], [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float16)
    expect = np.array([[[[1., 0., 0., 2.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [3., 0., 0., 4.]]]]).astype(np.float16)
    rnn = ResizeBilinearGradAlignCornerT()
    output = rnn(Tensor(dy), Tensor(orign_image))
    assert np.all(output.asnumpy() == expect)

    orign_image = np.array(
        [[[[1.1, 2.2, 3.2, 2.5], [3.3, 4.4, 5.7, 8.1], [3.3, 4.4, 5.7, 8.1], [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float32)
    expect = np.array([[[[1., 0., 0., 2.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [3., 0., 0., 4.]]]]).astype(np.float32)
    rnn = ResizeBilinearGradAlignCornerT()
    output = rnn(Tensor(dy), Tensor(orign_image))
    assert np.all(output.asnumpy() == expect)


def test_resize_bilinear_grad_align_corner_false():
    """
    Feature: Test ResizeBilinearGrad on CPU.
    Description:  Test align corner false.
    Expectation: Assert that results are consistent with expect.
    """
    dy = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]]).astype(np.float32)

    orign_image = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(np.float16)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(np.float16)
    rnn = ResizeBilinearGradAlignCornerF()
    output = rnn(Tensor(dy), Tensor(orign_image))
    assert np.all(output.asnumpy() == expect)

    orign_image = np.array([[[[1.1, 2.2], [3.3, 4.4]]]]).astype(np.float32)
    expect = np.array([[[[2.25, 0.75],
                         [0.75, 4.25]]]]).astype(np.float32)
    rnn = ResizeBilinearGradAlignCornerF()
    output = rnn(Tensor(dy), Tensor(orign_image))
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_resize_bilinear_grad_dtype(mode, dtype):
    """
    Feature: Test ResizeBilinearGrad on CPU.
    Description:  Test float16, float32, float64.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=mode, device_target="CPU")
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(dtype)
    orign_image = np.array(
        [[[[1.1, 2.2, 3.2, 2.5], [3.3, 4.4, 5.7, 8.1], [3.3, 4.4, 5.7, 8.1], [3.3, 4.4, 5.7, 8.1]]]]).astype(dtype)
    expect = np.array([[[[1., 0., 0., 2.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [3., 0., 0., 4.]]]]).astype(dtype)
    rnn = ResizeBilinearGradAlignCornerT()
    output = rnn(Tensor(dy), Tensor(orign_image))
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu_training
def test_resize_bilinear_grad_half_pixel_centers():
    """
    Feature: Test ResizeBilinearGrad on CPU.
    Description:  The half_pixel_centers is True.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dy = np.array([[[[1, 2], [3, 4]]]]).astype(np.float16)

    x = np.array([[[[1.1, 2.2, 3.2, 2.5],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1],
                    [3.3, 4.4, 5.7, 8.1]]]]).astype(np.float16)
    expect = np.array([[[[0.25, 0.25, 0.5, 0.5],
                         [0.25, 0.25, 0.5, 0.5],
                         [0.75, 0.75, 1.0, 1.0],
                         [0.75, 0.75, 1.0, 1.0]]]], dtype=np.float16)
    net = NetResizeBilinearFunc(half_pixel_centers=True)
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
    net = NetResizeBilinearFunc(half_pixel_centers=True)
    output = net(Tensor(dy), Tensor(x))
    assert np.all(output.asnumpy() == expect)
    