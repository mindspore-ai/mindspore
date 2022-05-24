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
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context, Parameter

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class ApplyProximalAdagradTEST(nn.Cell):
    def __init__(self):
        super(ApplyProximalAdagradTEST, self).__init__()
        self.apply_proximal_adagrad = P.ApplyProximalAdagrad()
        self.var = Parameter(
            Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)), name="var")
        self.accum = Parameter(
            Tensor(np.array([[0.6, 0.5], [0.2, 0.6]]).astype(np.float32)), name="accum")

    def construct(self, lr, l1, l2, grad):
        return self.apply_proximal_adagrad(self.var, self.accum, lr, l1, l2, grad)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_apply_proximal_adagrad_op(data_type):
    """
    Feature: ApplyProximalAdagrad cpu kernel
    Description: test the ApplyProximalAdagrad.
    Expectation: match to np benchmark.
    """
    adgrad = ApplyProximalAdagradTEST()
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3

    lr = 0.01
    l1 = 0.0
    l2 = 0.0
    grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))

    context.set_context(mode=context.GRAPH_MODE)
    output = adgrad(lr, l1, l2, grad)
    mindspore_var_out = output[0].asnumpy()
    mindspore_accum_out = output[1].asnumpy()
    print(mindspore_var_out)
    print(mindspore_accum_out)

    expect_var = np.array([[5.96388459e-01, 3.92964751e-01], [9.78178233e-02, 4.92815793e-01]]).astype(data_type)
    expect_accum = np.array([[6.90000057e-01, 9.90000010e-01], [2.10000008e-01, 1.24000001e+00]]).astype(data_type)

    np.testing.assert_allclose(mindspore_var_out, expect_var, rtol=error)
    np.testing.assert_allclose(mindspore_accum_out, expect_accum, rtol=error)
