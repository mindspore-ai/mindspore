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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        out_channel = 64
        kernel_size = 7
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)
        self.w = Parameter(initializer(
            'normal', [64, 3, 7, 7]), name='w')

    def construct(self, x):
        return self.conv(x, self.w)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=812.0)
def test_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.random.randn(32, 3, 224, 224).astype(np.float32)
    conv = Net()
    output = conv(Tensor(x))
    assert output.shape == (32, 64, 218, 218)
