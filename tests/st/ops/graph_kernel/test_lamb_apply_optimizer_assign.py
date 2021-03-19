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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P



class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.lamb_apply_optimizer_assign = P.LambApplyOptimizerAssign()

    def construct(self, grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon,
                  steps, do_use_weight, weight_decay_rate):
        return self.lamb_apply_optimizer_assign(grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2,
                                                one_minus_beta_2, epsilon, steps, do_use_weight, weight_decay_rate)

def get_output(grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon, steps,
               do_use_weight, weight_decay_rate, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    opt = Net()
    output = opt(Tensor(grad), Tensor(inputv), Tensor(inputm), Tensor(input_param), Tensor(beta_1),
                 Tensor(one_minus_beta_1), Tensor(beta_2), Tensor(one_minus_beta_2), Tensor(epsilon), Tensor(steps),
                 Tensor(do_use_weight), Tensor(weight_decay_rate))
    return output

def lamb_apply_optimizer_assign():

    grad = np.array([0.01, 0.03, 0.05]).astype(np.float32)
    inputv = np.array([1.2, 3.4, 5.6]).astype(np.float32)
    inputm = np.array([0.11, 0.33, 0.55]).astype(np.float32)
    input_param = np.array([1, 3, 5]).astype(np.float32)
    beta_1 = np.array([0.9]).astype(np.float32)
    beta_2 = np.array([0.999]).astype(np.float32)
    one_minus_beta_1 = (np.array([1.0]) - np.array([0.9])).astype(np.float32)
    one_minus_beta_2 = (np.array([1.0]) - np.array([0.999])).astype(np.float32)
    epsilon = np.array([1e-6]).astype(np.float32)
    steps = np.array([10]).astype(np.float32)
    do_use_weight = np.array([1]).astype(np.float32)
    weight_decay_rate = np.array([0.021]).astype(np.float32)

    expect = get_output(grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon,
                        steps, do_use_weight, weight_decay_rate, False)
    output = get_output(grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon,
                        steps, do_use_weight, weight_decay_rate, True)

    e1, e2, e3 = list(expect)
    o1, o2, o3 = list(output)

    assert np.allclose(o1.asnumpy(), e1.asnumpy())
    assert np.allclose(o2.asnumpy(), e2.asnumpy())
    assert np.allclose(o3.asnumpy(), e3.asnumpy())

def test_lamb_apply_optimizer_assign_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    lamb_apply_optimizer_assign()
