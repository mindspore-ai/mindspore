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
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import numpy as np
import pytest


class ApplyAdamWithAmsgradV2Net(nn.Cell):
    def __init__(self, use_locking=False):
        super(ApplyAdamWithAmsgradV2Net, self).__init__()
        self.apply_adam_with_amsgrad = ops.ApplyAdamWithAmsgradV2(use_locking)
        self.var = Parameter(Tensor(np.array([[0.2, 0.2], [0.2, 0.2]]).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.array([[0.1, 0.2], [0.4, 0.3]]).astype(np.float32)), name="m")
        self.v = Parameter(Tensor(np.array([[0.2, 0.1], [0.3, 0.4]]).astype(np.float32)), name="v")
        self.vhat = Parameter(Tensor(np.array([[0.1, 0.2], [0.6, 0.2]]).astype(np.float32)), name="vhat")
        self.beta1 = 0.8
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.beta1_power = 0.9
        self.beta2_power = 0.999
        self.lr = 0.01

    def construct(self, grad):
        out = self.apply_adam_with_amsgrad(self.var, self.m, self.v, self.vhat,
                                           self.beta1_power, self.beta2_power, self.lr, self.beta1, self.beta2,
                                           self.epsilon, grad)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_applyadamwithamsgradv2(mode):
    """
    Feature: ops.ApplyAdamWithAmsgradV2
    Description: Verify the result of ApplyAdamWithAmsgradV2
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = ApplyAdamWithAmsgradV2Net()
    grad = Tensor(np.array([[0.4, 0.2], [0.2, 0.3]]).astype(np.float32))
    net(grad)
    expect_output = np.array([[0.19886853, 0.1985858], [0.19853032, 0.19849943]])
    assert np.allclose(net.var.asnumpy(), expect_output)
