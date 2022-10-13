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
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.nn import layer
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops.operations import _inner_ops as inner

num_one = Tensor(np.ones([1]), mstype.float32)
data_type = np.float32
param_val = np.array([[1, 1, 2], [3, 4, 5], [5, 6, 7]]).astype(data_type)
m_val = np.array([[1, 1, 2], [3, 4, 5], [5, 6, 7]]).astype(data_type)
v_val = np.array([[1, 1, 2], [3, 4, 5], [5, 6, 7]]).astype(data_type)
grad_val = np.array([[1, 1, 2], [3, 4, 5], [5, 6, 7]]).astype(data_type)

beta1_val = Tensor(np.array([0.9]).astype(data_type))
beta2_val = Tensor(np.array([0.999]).astype(data_type))
eps_val = Tensor(np.array([1e-6]).astype(data_type))
global_step_val = Tensor(np.array([2]).astype(np.int32))
lr_val = Tensor(np.array([0.01]).astype(data_type))
weight_decay_val = Tensor(np.array([0.0]).astype(data_type))


class LambGPUOrigin(nn.Cell):
    def __init__(self, param, m, v, gradient):
        super().__init__()
        self.param = Parameter(param, name="param")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")
        self.gradient = Parameter(gradient, name="grad")
        self.op_mul = P.Mul()
        self.op_sqrt = P.Sqrt()
        self.op_rsqrt = P.Rsqrt()
        self.op_square = P.Square()
        self.op_cast = P.Cast()
        self.op_reshape = P.Reshape()
        self.op_shape = P.Shape()
        self.op_pow = P.Pow()
        self.op_norm = layer.Norm()
        self.op_select = P.Select()
        self.op_greater = P.Greater()
        self.op_fill = P.Fill()
        self.op_dtype = P.DType()

    def construct(self, beta1, beta2, eps, global_step, lr, weight_decay, decay_flag):
        param_fp32 = self.op_cast(self.param, mstype.float32)
        m_fp32 = self.op_cast(self.m, mstype.float32)
        v_fp32 = self.op_cast(self.v, mstype.float32)
        gradient_fp32 = self.op_cast(self.gradient, mstype.float32)

        next_m = self.op_mul(beta1, m_fp32) + self.op_mul(self.op_cast(num_one, mstype.float32) - beta1, gradient_fp32)

        next_v = self.op_mul(beta2, v_fp32) + self.op_mul(self.op_cast(num_one, mstype.float32) - beta2,
                                                          self.op_square(gradient_fp32))

        next_mm = next_m / (self.op_cast(num_one, mstype.float32) -
                            self.op_pow(beta1, self.op_cast(global_step, mstype.float32)))
        next_vv = next_v / (self.op_cast(num_one, mstype.float32) -
                            self.op_pow(beta2, self.op_cast(global_step, mstype.float32)))
        w_norm = self.op_norm(param_fp32)
        g_norm = self.op_norm(gradient_fp32)

        g_norm_hat = self.op_norm(self.op_mul(next_mm, self.op_rsqrt(next_vv + eps)) + weight_decay * param_fp32)
        zeros = F.zeros_like(w_norm)
        ones = self.op_fill(self.op_dtype(w_norm), self.op_shape(w_norm), 1.0)
        trust_ratio = self.op_select(
            self.op_greater(w_norm, zeros),
            self.op_select(self.op_greater(g_norm, zeros), w_norm / g_norm_hat, ones),
            ones)
        tens = self.op_fill(self.op_dtype(trust_ratio), self.op_shape(trust_ratio), 10.0)
        trust_ratio = C.clip_by_value(trust_ratio, zeros, tens)
        update = next_mm / (self.op_sqrt(next_vv) + eps)

        if decay_flag:
            update = update + self.op_mul(weight_decay, param_fp32)

        update_with_lr = self.op_mul(self.op_mul(trust_ratio, lr), update)

        next_param = param_fp32 - self.op_reshape(update_with_lr, self.op_shape(param_fp32))

        F.assign(self.param, self.op_cast(next_param, F.dtype(self.param)))
        F.assign(self.m, self.op_cast(next_m, F.dtype(self.m)))
        F.assign(self.v, self.op_cast(next_v, F.dtype(self.v)))

        return self.op_cast(next_param, F.dtype(self.param))


class LambAscendOrigin(nn.Cell):
    def __init__(self, param, m, v, gradient):
        super().__init__()
        self.param = Parameter(param, name="param")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")
        self.gradient = Parameter(gradient, name="grad")
        self.op_lamb_apply_optimizer_assign = P.LambApplyOptimizerAssign()
        self.op_lamb_apply_weight_assign = P.LambApplyWeightAssign()
        self.op_cast = P.Cast()
        self.op_norm = layer.Norm()

    def construct(self, beta1, beta2, eps, global_step, lr, weight_decay, decay_flag):
        param_fp32 = self.op_cast(self.param, mstype.float32)
        gradient_fp32 = self.op_cast(self.gradient, mstype.float32)
        new_global_step = self.op_cast(global_step, mstype.float32)
        weight_decay_flag = self.op_cast(decay_flag, mstype.float32)

        update, _, _ = self.op_lamb_apply_optimizer_assign(gradient_fp32, self.v, self.m, param_fp32,
                                                           beta1, 1.0 - beta1, beta2, 1.0 - beta2, eps,
                                                           new_global_step, weight_decay_flag, weight_decay)
        w_norm = self.op_norm(param_fp32)
        g_norm = self.op_norm(update)

        update = F.depend(update, self.op_lamb_apply_weight_assign(w_norm, g_norm, lr, update, self.param))
        return self.param


class MyLamb(nn.Cell):
    def __init__(self, param, m, v, gradient):
        super().__init__()
        self.param = Parameter(param, name="param")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")
        self.gradient = Parameter(gradient, name="grad")
        self.lamb = inner.Lamb()

    def construct(self, beta1, beta2, eps, global_step, lr, weight_decay):
        return self.lamb(self.param, self.m, self.v, lr, beta1, beta2, eps, weight_decay, global_step, self.gradient)


def test_gpu_net():
    """
    Feature: gpu testcase for Lamb
    Description: fixed input when using gpu
    Expectation: get the same result when use new lamb kernel and old kernel
    """
    my_lamb = MyLamb(param_val, m_val, v_val, grad_val)
    my_lamb(beta1_val, beta2_val, eps_val, global_step_val, lr_val, weight_decay_val)

    lamb_gpu_origin = LambGPUOrigin(param_val, m_val, v_val, grad_val)
    lamb_gpu_origin(beta1_val, beta2_val, eps_val, global_step_val, lr_val, weight_decay_val, True)

    assert np.allclose(my_lamb.param.asnumpy(), lamb_gpu_origin.param.asnumpy())


def test_ascend_net():
    """
    Feature: ascend testcase for Lamb
    Description: fixed input when using ascend
    Expectation: get the same result when use new lamb kernel and old kernel
    """
    my_lamb = MyLamb(param_val, m_val, v_val, grad_val)
    my_lamb(beta1_val, beta2_val, eps_val, global_step_val, lr_val, weight_decay_val)

    lamb_ascend_origin = LambAscendOrigin(param_val, m_val, v_val, grad_val)
    lamb_ascend_origin(beta1_val, beta2_val, eps_val, global_step_val, lr_val, weight_decay_val, True)

    assert np.allclose(my_lamb.param.asnumpy(), lamb_ascend_origin.param.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_graph_net():
    """
    Feature: graph kernel testcase for Lamb
    Description: fixed input when using ascend in graph mode
    Expectation: get the same result when use new lamb kernel and old kernel
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_gpu_net()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_pynative_net():
    """
    Feature: pynative kernel testcase for Lamb
    Description: fixed input when using ascend in pynative mode
    Expectation: get the same result when use new lamb kernel and old kernel
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_gpu_net()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_graph_net():
    """
    Feature: graph kernel testcase for Lamb
    Description: fixed input when using ascend in graph mode
    Expectation: get the same result when use new lamb kernel and old kernel
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_ascend_net()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_pynative_net():
    """
    Feature: pynative kernel testcase for Lamb
    Description: fixed input when using ascend in pynative mode
    Expectation: get the same result when use new lamb kernel and old kernel
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_ascend_net()
