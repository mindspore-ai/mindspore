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
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=True)


class Net(nn.Cell):
    def __init__(self, decay_flag=True):
        super(Net, self).__init__()
        self.decay_flag = decay_flag
        self.op_mul = P.Mul()
        self.op_square = P.Square()
        self.op_sqrt = P.Sqrt()
        self.op_cast = P.Cast()
        self.op_reshape = P.Reshape()
        self.op_shape = P.Shape()
        self.param = Parameter(Tensor(np.array([0.1, 0.3, 0.5]).astype(np.float32)), name='param')
        self.m = Parameter(Tensor(np.array([0.1, 0.3, 0.5]).astype(np.float32)), name='m')
        self.v = Parameter(Tensor(np.array([0.1, 0.3, 0.5]).astype(np.float32)), name='v')

    @ms_function
    def construct(self, beta1, beta2, gradient, eps, weight_decay_tensor, lr):
        param_fp32 = self.op_cast(self.param, mstype.float32)
        m_fp32 = self.op_cast(self.m, mstype.float32)
        v_fp32 = self.op_cast(self.v, mstype.float32)
        gradient_fp32 = self.op_cast(gradient, mstype.float32)

        next_m = self.op_mul(beta1, m_fp32) + \
                 self.op_mul(self.op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta1, gradient_fp32)
        next_v = self.op_mul(beta2, v_fp32) + self.op_mul(self.op_cast(F.tuple_to_array((1.0,)), mstype.float32) - \
                                                          beta2, self.op_square(gradient_fp32))
        update = next_m / (eps + self.op_sqrt(next_v))
        if self.decay_flag:
            update = self.op_mul(weight_decay_tensor, param_fp32) + update
        update_with_lr = self.op_mul(lr, update)
        next_param = param_fp32 - self.op_reshape(update_with_lr, self.op_shape(param_fp32))

        next_v = F.depend(next_v, F.assign(self.param, next_param))
        next_v = F.depend(next_v, F.assign(self.m, next_m))
        next_v = F.depend(next_v, F.assign(self.v, next_v))
        return next_v


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adam_fusion():
    beta1 = Tensor(np.array([0.9]).astype(np.float32))
    beta2 = Tensor(np.array([0.999]).astype(np.float32))
    lr = Tensor(np.array([0.001]).astype(np.float32))
    eps = Tensor(np.array([1e-6]).astype(np.float32))
    weight_decay_tensor = Tensor(np.array([0.001]).astype(np.float32))

    gradient = Tensor(np.array([0.01, 0.03, 0.05]).astype(np.float32))
    opt = Net(True)
    _ = opt(beta1, beta2, gradient, eps, weight_decay_tensor, lr)

    param_expect = np.array([0.09971199, 0.29950103, 0.4993557]).astype(np.float32)
    m_expect = np.array([0.091, 0.273, 0.45499998]).astype(np.float32)
    v_expect = np.array([0.0999001, 0.29970092, 0.4995025]).astype(np.float32)
    assert np.allclose(opt.param.data.asnumpy(), param_expect)
    assert np.allclose(opt.m.data.asnumpy(), m_expect)
    assert np.allclose(opt.v.data.asnumpy(), v_expect)
