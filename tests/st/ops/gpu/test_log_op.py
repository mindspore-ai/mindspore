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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context


class NetLog(nn.Cell):
    def __init__(self):
        super(NetLog, self).__init__()
        self.log = P.Log()

    def construct(self, x):
        return self.log(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_log():
    x0_np = np.random.uniform(1, 2, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.uniform(1, 2, 1).astype(np.float32)
    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)
    expect0 = np.log(x0_np)
    expect1 = np.log(x1_np)
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    error1 = np.ones(shape=expect1.shape) * 1.0e-5

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    log = NetLog()
    output0 = log(x0)
    output1 = log(x1)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert (output0.shape() == expect0.shape)
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert (output1.shape() == expect1.shape)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    log = NetLog()
    output0 = log(x0)
    output1 = log(x1)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert (output0.shape() == expect0.shape)
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert (output1.shape() == expect1.shape)
