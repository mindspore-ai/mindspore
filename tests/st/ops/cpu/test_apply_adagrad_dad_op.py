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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class ApplyAdagradDANet(nn.Cell):
    def __init__(self, use_locking=False):
        super(ApplyAdagradDANet, self).__init__()
        self.apply_adagrad_d_a = P.ApplyAdagradDA(use_locking)
        self.var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)), name="var")
        self.gradient_accumulator = Parameter(Tensor(np.array([[0.1, 0.3],
                                                               [0.1, 0.5]]).astype(np.float32)),
                                              name="gradient_accumulator")
        self.gradient_squared_accumulator = Parameter(Tensor(np.array([[0.2, 0.1],
                                                                       [0.1, 0.2]]).astype(np.float32)),
                                                      name="gradient_squared_accumulator")

    def construct(self, grad, lr, l1, l2, global_step):
        out = self.apply_adagrad_d_a(self.var, self.gradient_accumulator,
                                     self.gradient_squared_accumulator, grad, lr, l1, l2, global_step)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_apply_adagrad_da():
    """
    Feature: ApplyAdagradDAD
    Description: Test the calculation difference between numpy and mindscore in ApplyAdagradDA
    Expectation: success
    """
    # numpy
    np_grad_accum = np.array([[0.1, 0.3],
                              [0.1, 0.5]])
    np_grad_squared_accum = np.array([[0.2, 0.1],
                                      [0.1, 0.2]])
    np_grad = np.array([[0.3, 0.4],
                        [0.1, 0.2]])
    np_lr = 0.001
    np_l1 = 0.001
    np_l2 = 0.001
    np_global_step = 2
    np_grad_accum += np_grad
    np_grad_squared_accum += np_grad * np_grad
    tmp_val = np.sign(np_grad_accum) * np.maximum(np.abs(np_grad_accum) - np_l1 * np_global_step,
                                                  0) if np_l1 > 0 else np_grad_accum
    x_value = -1 * np_lr * tmp_val
    y_value = np_l2 * np_global_step * np_lr + np.sqrt(np_grad_squared_accum)
    np_var = x_value / y_value
    # mindspore
    net = ApplyAdagradDANet()
    grad = Tensor(np.array([[0.3, 0.4], [0.1, 0.2]]).astype(np.float32))
    lr = Tensor(0.001, mstype.float32)
    l1 = Tensor(0.001, mstype.float32)
    l2 = Tensor(0.001, mstype.float32)
    global_step = Tensor(2, mstype.int32)
    res_var_mindspore = net(grad, lr, l1, l2, global_step)
    eps = np.array([1e-6 for i in range(4)]).reshape(2, 2)
    assert np.all(np_var - res_var_mindspore < eps)
