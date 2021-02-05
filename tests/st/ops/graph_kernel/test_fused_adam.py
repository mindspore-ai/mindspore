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
        self.param = Parameter(
            Tensor(np.array([1, 3, 5]).astype(np.float32)), name='param')
        self.m = Parameter(
            Tensor(np.array([0.11, 0.33, 0.55]).astype(np.float32)), name='m')
        self.v = Parameter(
            Tensor(np.array([1.2, 3.4, 5.6]).astype(np.float32)), name='v')

    @ms_function
    def construct(self, beta1, beta2, one_sub_beta_1, one_sub_beta_2, gradient, eps, weight_decay_tensor, lr):
        param_fp32 = self.op_cast(self.param, mstype.float32)
        m_fp32 = self.op_cast(self.m, mstype.float32)
        v_fp32 = self.op_cast(self.v, mstype.float32)
        gradient_fp32 = self.op_cast(gradient, mstype.float32)

        next_m = self.op_mul(beta1, m_fp32) + \
            self.op_mul(self.op_cast(one_sub_beta_1,
                                     mstype.float32), gradient_fp32)
        next_v = self.op_mul(beta2, v_fp32) + self.op_mul(self.op_cast(one_sub_beta_2,
                                                                       mstype.float32), self.op_square(gradient_fp32))
        update = next_m / (eps + self.op_sqrt(next_v))
        if self.decay_flag:
            update = self.op_mul(weight_decay_tensor, param_fp32) + update
        update_with_lr = self.op_mul(lr, update)
        next_param = param_fp32 - \
            self.op_reshape(update_with_lr, self.op_shape(param_fp32))

        depend_v = F.depend(next_param, F.assign(self.param, next_param))
        depend_v = F.depend(depend_v, F.assign(self.m, next_m))
        depend_v = F.depend(depend_v, F.assign(self.v, next_v))
        return depend_v


class SideEffectFusedAdamNet(nn.Cell):
    def __init__(self, decay_flag=True):
        super(SideEffectFusedAdamNet, self).__init__()
        self.decay_flag = decay_flag
        self.op_mul = P.Mul()
        self.op_square = P.Square()
        self.op_sqrt = P.Sqrt()
        self.op_cast = P.Cast()
        self.op_reshape = P.Reshape()
        self.op_shape = P.Shape()
        self.param = Parameter(
            Tensor(np.array([0, 0, 0]).astype(np.float32)), name='param')
        self.m = Parameter(
            Tensor(np.array([0.11, 0.33, 0.55]).astype(np.float32)), name='m')
        self.v = Parameter(
            Tensor(np.array([1.2, 3.4, 5.6]).astype(np.float32)), name='v')
        self.x = Parameter(
            Tensor(np.array([1, 3, 5]).astype(np.float32)), name='x')

    @ms_function
    def construct(self, beta1, beta2, one_sub_beta_1, one_sub_beta_2, gradient, eps, weight_decay_tensor, lr):
        F.assign(self.param, self.x)

        param_fp32 = self.op_cast(self.param, mstype.float32)
        m_fp32 = self.op_cast(self.m, mstype.float32)
        v_fp32 = self.op_cast(self.v, mstype.float32)
        gradient_fp32 = self.op_cast(gradient, mstype.float32)

        next_m = self.op_mul(beta1, m_fp32) + \
            self.op_mul(self.op_cast(one_sub_beta_1,
                                     mstype.float32), gradient_fp32)
        next_v = self.op_mul(beta2, v_fp32) + self.op_mul(self.op_cast(one_sub_beta_2,
                                                                       mstype.float32), self.op_square(gradient_fp32))
        update = next_m / (eps + self.op_sqrt(next_v))
        if self.decay_flag:
            update = self.op_mul(weight_decay_tensor, param_fp32) + update
        update_with_lr = self.op_mul(lr, update)
        next_param = param_fp32 - \
            self.op_reshape(update_with_lr, self.op_shape(param_fp32))

        depend_v = F.depend(next_param, F.assign(self.param, next_param))
        depend_v = F.depend(depend_v, F.assign(self.m, next_m))
        depend_v = F.depend(depend_v, F.assign(self.v, next_v))

        F.assign(self.x, self.m)
        return depend_v


def CalFusedAdam(beta1, beta2, one_sub_beta_1, one_sub_beta_2, gradient, eps, weight_decay_tensor, lr, param, m, v,
                 is_weight_decay=False):
    m_expect = beta1 * m + one_sub_beta_1 * gradient
    v_expect = beta2 * v + one_sub_beta_2 * gradient * gradient
    update = m_expect / (np.sqrt(v_expect) + eps)
    if is_weight_decay:
        update += weight_decay_tensor * param
    param_expect = param - lr * update
    return param_expect, m_expect, v_expect


def test_adam():
    np.random.seed(0)
    beta1 = np.array([0.9]).astype(np.float32)
    beta2 = np.array([0.999]).astype(np.float32)
    one_sub_beta_1 = (np.array([1.0]) - np.array([0.9])).astype(np.float32)
    one_sub_beta_2 = (np.array([1.0]) - np.array([0.999])).astype(np.float32)
    lr = np.array([0.012]).astype(np.float32)
    eps = np.array([1e-6]).astype(np.float32)
    weight_decay_tensor = np.array([0.021]).astype(np.float32)

    gradient = np.array([0.01, 0.03, 0.05]).astype(np.float32)
    m = np.array([0.11, 0.33, 0.55]).astype(np.float32)
    v = np.array([1.2, 3.4, 5.6]).astype(np.float32)
    param = np.array([1, 3, 5]).astype(np.float32)
    is_weight_decay = False
    opt = Net(is_weight_decay)
    _ = opt(Tensor(beta1), Tensor(beta2), Tensor(one_sub_beta_1), Tensor(one_sub_beta_2), Tensor(gradient), Tensor(eps),
            Tensor(weight_decay_tensor), Tensor(lr))
    param_expect, m_expect, v_expect = CalFusedAdam(
        beta1, beta2, one_sub_beta_1, one_sub_beta_2, gradient, eps, weight_decay_tensor, lr,
        param, m, v, is_weight_decay)
    assert np.allclose(opt.param.data.asnumpy(), param_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.m.data.asnumpy(), m_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.v.data.asnumpy(), v_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)


def test_adam_weight_decay():
    np.random.seed(0)
    beta1 = np.array([0.9]).astype(np.float32)
    beta2 = np.array([0.999]).astype(np.float32)
    one_sub_beta_1 = (np.array([1.0]) - np.array([0.9])).astype(np.float32)
    one_sub_beta_2 = (np.array([1.0]) - np.array([0.999])).astype(np.float32)
    lr = np.array([0.012]).astype(np.float32)
    eps = np.array([1e-6]).astype(np.float32)
    weight_decay_tensor = np.array([0.021]).astype(np.float32)

    gradient = np.array([0.01, 0.03, 0.05]).astype(np.float32)
    m = np.array([0.11, 0.33, 0.55]).astype(np.float32)
    v = np.array([1.2, 3.4, 5.6]).astype(np.float32)
    param = np.array([1, 3, 5]).astype(np.float32)
    is_weight_decay = True
    opt = Net(is_weight_decay)
    _ = opt(Tensor(beta1), Tensor(beta2), Tensor(one_sub_beta_1), Tensor(one_sub_beta_2), Tensor(gradient), Tensor(eps),
            Tensor(weight_decay_tensor), Tensor(lr))
    param_expect, m_expect, v_expect = CalFusedAdam(
        beta1, beta2, one_sub_beta_1, one_sub_beta_2, gradient, eps, weight_decay_tensor, lr,
        param, m, v, is_weight_decay)

    assert np.allclose(opt.param.data.asnumpy(), param_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.m.data.asnumpy(), m_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.v.data.asnumpy(), v_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)


def test_adam_side_effect():
    np.random.seed(0)
    beta1 = np.array([0.9]).astype(np.float32)
    beta2 = np.array([0.999]).astype(np.float32)
    one_sub_beta_1 = (np.array([1.0]) - np.array([0.9])).astype(np.float32)
    one_sub_beta_2 = (np.array([1.0]) - np.array([0.999])).astype(np.float32)
    lr = np.array([0.012]).astype(np.float32)
    eps = np.array([1e-6]).astype(np.float32)
    weight_decay_tensor = np.array([0.021]).astype(np.float32)

    gradient = np.array([0.01, 0.03, 0.05]).astype(np.float32)
    m = np.array([0.11, 0.33, 0.55]).astype(np.float32)
    v = np.array([1.2, 3.4, 5.6]).astype(np.float32)
    param = np.array([1, 3, 5]).astype(np.float32)
    is_weight_decay = False
    opt = SideEffectFusedAdamNet(is_weight_decay)
    _ = opt(Tensor(beta1), Tensor(beta2), Tensor(one_sub_beta_1), Tensor(one_sub_beta_2), Tensor(gradient), Tensor(eps),
            Tensor(weight_decay_tensor), Tensor(lr))
    param_expect, m_expect, v_expect = CalFusedAdam(
        beta1, beta2, one_sub_beta_1, one_sub_beta_2, gradient, eps, weight_decay_tensor, lr,
        param, m, v, is_weight_decay)
    assert np.allclose(opt.param.data.asnumpy(), param_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.m.data.asnumpy(), m_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.v.data.asnumpy(), v_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(opt.x.data.asnumpy(), m_expect,
                       rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adam_gpu():
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    test_adam()


def test_adam_ascend():
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="Ascend")
    test_adam()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adam_weight_decay_gpu():
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    test_adam_weight_decay()


def test_adam_weight_decay_ascend():
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="Ascend")
    test_adam_weight_decay()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adam_side_effect_gpu():
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    test_adam_side_effect()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_adam_side_effect_ascend():
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="Ascend")
    test_adam_side_effect()
