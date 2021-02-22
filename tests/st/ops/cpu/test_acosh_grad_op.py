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
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetAcoshGrad(nn.Cell):
    def __init__(self):
        super(NetAcoshGrad, self).__init__()
        self.acoshGrad = G.AcoshGrad()

    def construct(self, x, dy):
        return self.acoshGrad(x, dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_acosh_grad():
    x = np.array([5, 4, 3]).astype('float32')
    dy = np.array([1, 0, -1]).astype('float32')
    acosh_grad = NetAcoshGrad()
    output = acosh_grad(Tensor(x), Tensor(dy))
    print(output)
    expect = dy / np.sqrt(x * x - 1)
    assert np.allclose(output.asnumpy(), expect)
