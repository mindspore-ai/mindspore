# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

param_shape = [2, 3, 2]


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.opt = ops.FusedAdaFactor()
        self.param = Parameter(Tensor(np.ones(param_shape), mstype.float32), name="param")
        self.exp_avg = Parameter(Tensor(np.zeros(param_shape), mstype.float32), name="exp_avg")
        self.exp_avg_sq = Parameter(Tensor(np.zeros(param_shape), mstype.float32), name="exp_avg_sq")
        self.exp_avg_sq_row = Parameter(Tensor(np.zeros([2, 3]), mstype.float32), name="exp_avg_sq_row")
        self.exp_avg_sq_col = Parameter(Tensor(np.zeros([2, 2]), mstype.float32), name="exp_avg_sq_col")

    def construct(self, epsilon, clip_threshold, beta1, beta2, weight_decay, lr, grad):
        out = self.opt(epsilon, clip_threshold, beta1, beta2, weight_decay, lr, grad, self.param, self.exp_avg,
                       self.exp_avg_sq_row, self.exp_avg_sq_col, self.exp_avg_sq)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adafactor():
    '''
    Feature: AdaFactor
    Description: Test AdaFactor
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net()
    gradient = Tensor(np.ones(param_shape), mstype.float32)
    net((1e-30, 1e-3), 1.0, 0.9, 0.8, 1e-2, 0.03, gradient)
    diff = net.param.asnumpy() - np.ones(param_shape) * 0.97
    assert np.all(diff < 1e-3)
