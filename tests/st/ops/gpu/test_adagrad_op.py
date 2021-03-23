# Copyright 2020 Huawei Technologies Co., Ltd
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

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

var_np = np.random.rand(3, 3).astype(np.float32)
accum_np = np.random.rand(3, 3).astype(np.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_adagrad = P.ApplyAdagrad()
        self.var = Parameter(Tensor(var_np), name="var")
        self.accum = Parameter(Tensor(accum_np), name="accum")

    def construct(self, lr, grad):
        z = self.apply_adagrad(self.var, self.accum, lr, grad)
        return z


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_adagrad():
    # numpy op
    grident_np = np.random.rand(3, 3).astype(np.float32)
    expect_accum_np = accum_np + grident_np * grident_np
    expect_var_np = var_np - (0.001 * grident_np * (1 / np.sqrt(expect_accum_np + 1e-6)))

    net = Net()
    lr = Tensor(0.001, mstype.float32)
    grad = Tensor(grident_np)
    out = net(lr, grad)
    res_var_mindspore = out[0].asnumpy()
    res_accum_mindspore = out[1].asnumpy()
    eps = np.array([1e-6 for i in range(9)]).reshape(3, 3)

    assert np.all(expect_var_np - res_var_mindspore < eps)
    assert np.all(expect_accum_np - res_accum_mindspore < eps)
