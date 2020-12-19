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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class ResizeBilinearGradAlignCornerT(nn.Cell):
    def __init__(self):
        super(ResizeBilinearGradAlignCornerT, self).__init__()
        self.ResizeBilinearGradAlignCornerT = G.ResizeBilinearGrad(
            align_corners=True)

    def construct(self, dy, size):
        return self.ResizeBilinearGradAlignCornerT(dy, size)


class ResizeBilinearGradAlignCornerF(nn.Cell):
    def __init__(self):
        super(ResizeBilinearGradAlignCornerF, self).__init__()
        self.ResizeBilinearGradAlignCornerF = G.ResizeBilinearGrad(align_corners=False)

    def construct(self, dy, size):
        return self.ResizeBilinearGradAlignCornerF(dy, size)


def test_ResizeBilinearGradAlignCornerT():
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


def test_ResizeBilinearGradAlignCornerF():
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
