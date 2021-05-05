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


class NetConv3d(nn.Cell):
    def __init__(self):
        super(NetConv3d, self).__init__()
        out_channel = 4
        kernel_size = 2
        self.conv = P.Conv3D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)

    def construct(self, x, w):
        return self.conv(x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d():
    x = Tensor(np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32))
    expect = np.array([[[[[12960., 13236.],
                          [13788., 14064.]],
                         [[15444., 15720.],
                          [16272., 16548.]]],
                        [[[32256., 33108.],
                          [34812., 35664.]],
                         [[39924., 40776.],
                          [42480., 43332.]]],
                        [[[51552., 52980.],
                          [55836., 57264.]],
                         [[64404., 65832.],
                          [68688., 70116.]]],
                        [[[70848., 72852.],
                          [76860., 78864.]],
                         [[88884., 90888.],
                          [94896., 96900.]]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetConv3d()
    output = net(x, w)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetConv3d()
    output = net(x, w)
    assert (output.asnumpy() == expect).all()
