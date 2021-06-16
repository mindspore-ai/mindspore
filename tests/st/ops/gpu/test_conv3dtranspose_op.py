# Copyright 2021 Huawei Technologies Co., Ltd
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


class NetConv3dTranspose(nn.Cell):
    def __init__(self):
        super(NetConv3dTranspose, self).__init__()
        in_channel = 2
        out_channel = 2
        kernel_size = 2
        self.conv_trans = P.Conv3DTranspose(in_channel, out_channel,
                                            kernel_size,
                                            pad_mode="pad",
                                            pad=1,
                                            stride=1,
                                            dilation=1,
                                            group=1)

    def construct(self, x, w):
        return self.conv_trans(x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_transpose():
    x = Tensor(np.arange(1 * 2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).astype(np.float32))
    w = Tensor(np.ones((2, 2, 2, 2, 2)).astype(np.float32))
    expect = np.array([[[[[320., 336.],
                          [368., 384.]],
                         [[464., 480.],
                          [512., 528.]]],
                        [[[320., 336.],
                          [368., 384.]],
                         [[464., 480.],
                          [512., 528.]]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    conv3dtranspose = NetConv3dTranspose()
    output = conv3dtranspose(x, w)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    conv3dtranspose = NetConv3dTranspose()
    output = conv3dtranspose(x, w)
    assert (output.asnumpy() == expect).all()
