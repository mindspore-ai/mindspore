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
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class Net(nn.Cell):
    def __init__(self, var_np, accum_np):
        super(Net, self).__init__()
        self.apply_addsign = P.ApplyAddSign()
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="m")

    def construct(self, lr, alpha, sign_decay, beta, grad):
        z = self.apply_addsign(self.var, self.accum, lr, alpha, sign_decay, beta, grad)
        return z


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_addsign_graph_float32():
    """
    Feature: ApplyAddSign gpu kernel.
    Description: test the ApplyAddSign.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    var_np = np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)
    accum_np = np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float32)
    grident_np = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32)
    expect_accum_np = 0.9 * accum_np + (1.0 - 0.9) * grident_np
    expect_update = (1.0 + 0.99 * np.sign(grident_np) * np.sign(expect_accum_np)) * grident_np
    expect_var_np = var_np - (0.001 * expect_update)
    net = Net(var_np, accum_np)
    lr = Tensor(0.001, mstype.float32)
    alpha = Tensor(1.0, mstype.float32)
    sign_decay = Tensor(0.99, mstype.float32)
    beta = Tensor(0.9, mstype.float32)
    grad = Tensor(grident_np)
    out = net(lr, alpha, sign_decay, beta, grad)
    res_var_mindspore = out[0].asnumpy()
    res_accum_mindspore = out[1].asnumpy()
    eps = np.array([1e-6 for i in range(4)]).reshape(2, 2)
    assert np.all(expect_var_np - res_var_mindspore < eps)
    assert np.all(expect_accum_np - res_accum_mindspore < eps)
