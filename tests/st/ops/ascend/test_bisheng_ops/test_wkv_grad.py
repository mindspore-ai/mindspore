# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations._grad_ops import WKVGrad

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.wkv_grad = WKVGrad()

    def construct(self, *args):
        return self.wkv_grad(*args)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_wkv_grad_float32_tensor_api():
    """
    Feature: test wkv tensor api.
    Description: test float32 inputs, need libwkv.so.
    Expectation: the result match with expected result.
    """
    b = 32
    t = 2
    c = 128
    w = Tensor(np.random.randn(c).astype(np.float32))
    u = Tensor(np.random.randn(c).astype(np.float32))
    k = Tensor(np.random.randn(b, t, c).astype(np.float32))
    v = Tensor(np.random.randn(b, t, c).astype(np.float32))
    gy = Tensor(np.random.randn(b, t, c).astype(np.float32))
    net = Net()
    backward = net(w, u, k, v, gy)
    print(backward)
