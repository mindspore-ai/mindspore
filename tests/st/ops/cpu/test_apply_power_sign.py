# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, Parameter
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_power_sign = ops.ApplyPowerSign()
        self.var = Parameter(Tensor(np.array([[0.6, 0.4],
                                              [0.1, 0.5]]).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.array([[0.6, 0.5],
                                            [0.2, 0.6]]).astype(np.float32)), name="m")
        self.lr = 0.001
        self.logbase = np.e
        self.sign_decay = 0.99
        self.beta = 0.9

    def construct(self, grad):
        out = self.apply_power_sign(self.var, self.m, self.lr, self.logbase,
                                    self.sign_decay, self.beta, grad)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_power_assign():
    """
    Feature: test ops ApplyPowerSign.
    Description: Update var and m by ApplyPowerSign op.
    Expectation: match to expected benchmark output.
    """
    grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
    net = Net()
    net(grad)
    expect_var = [[5.95575690e-01, 3.89676481e-01],
                  [9.85252112e-02, 4.88201708e-01]]
    expect_m = [[5.70000052e-01, 5.19999981e-01],
                [1.89999998e-01, 6.20000064e-01]]
    assert np.allclose(net.var.asnumpy(), expect_var, atol=0.0001, rtol=0.0001, equal_nan=True)
    assert np.allclose(net.m.asnumpy(), expect_m, atol=0.0001, rtol=0.0001, equal_nan=True)
