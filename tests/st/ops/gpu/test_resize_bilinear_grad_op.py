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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class ResizeBilinearGradNet(nn.Cell):
    def __init__(self, align_corners=False):
        super(ResizeBilinearGradNet, self).__init__()
        self.rb1 = G.ResizeBilinearGrad(align_corners=align_corners)

    def construct(self, dy, size):
        return self.rb1(dy, size)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bilinear_grad_align_corners():
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bilinear_grad():
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
