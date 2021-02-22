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

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetAtan(nn.Cell):
    def __init__(self):
        super(NetAtan, self).__init__()
        self.atan = P.Atan()

    def construct(self, x):
        return self.atan(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_atan():
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = NetAtan()
    output = net(input_x)
    print(output)
    expect = np.arctan(np_array)
    assert np.allclose(output.asnumpy(), expect)
