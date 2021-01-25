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

class NetElu(nn.Cell):
    def __init__(self):
        super(NetElu, self).__init__()
        self.elu = P.Elu()

    def construct(self, x):
        return self.elu(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_elu_fp16():
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float16))
    expect = np.array([[-0.632, 4.0, -0.999], [2.0, -0.993, 9.0]]).astype(np.float16)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    elu = NetElu()
    output = elu(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_elu_fp32():
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float32))
    expect = np.array([[-0.632, 4.0, -0.999], [2.0, -0.993, 9.0]]).astype(np.float32)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    elu = NetElu()
    output = elu(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
