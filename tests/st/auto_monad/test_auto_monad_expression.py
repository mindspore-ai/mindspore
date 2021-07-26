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
# ==============================================================================
import pytest
from mindspore.nn import Cell
from mindspore import context, Tensor, Parameter
import mindspore.ops.operations as P
import mindspore as ms
import numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class AutoMonadAddnAdamNet(Cell):
    def __init__(self, var, m, v):
        super().__init__()
        self.apply_adam = P.Adam()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")
        self.addn = P.AddN()
        self.mul = P.Mul()

    def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        out = self.addn((self.var, self.m, self.v))
        self.apply_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
        return out, self.var, self.m, self.v


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_monad_addn_adam():
    var = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    m = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    v = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    net = AutoMonadAddnAdamNet(var, m, v)
    beta1_power = Tensor(0.9, ms.float32)
    beta2_power = Tensor(0.999, ms.float32)
    lr = Tensor(0.1, ms.float32)
    beta1 = Tensor(0.9, ms.float32)
    beta2 = Tensor(0.999, ms.float32)
    epsilon = Tensor(1e-8, ms.float32)
    grad = Tensor(np.random.rand(3, 3, 3).astype(np.float32))
    out, new_var, new_m, new_v = net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    net = AutoMonadAddnAdamNet(var, m, v)
    context.set_context(mode=context.PYNATIVE_MODE)
    out_pyn, new_var_pyn, new_m_pyn, new_v_pyn = net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
    allclose_nparray(out_pyn.asnumpy(), out.asnumpy(), 0.001, 0.001)
    allclose_nparray(new_var_pyn.asnumpy(), new_var.asnumpy(), 0.001, 0.001)
    allclose_nparray(new_m_pyn.asnumpy(), new_m.asnumpy(), 0.001, 0.001)
    allclose_nparray(new_v_pyn.asnumpy(), new_v.asnumpy(), 0.001, 0.001)


class AutoMonadTwoAssignTwoAddnDependencyNet(Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([1.0], ms.float32), name="parameter1")
        self.parameter2 = ms.Parameter(Tensor([3.0], ms.float32), name="parameter2")
        self.assign = P.Assign()
        self.addN = P.AddN()

    def construct(self, inputs):
        self.assign(self.parameter1, inputs)
        out = self.addN((inputs, self.parameter1, self.parameter2))
        self.assign(self.parameter2, inputs)
        out = self.addN((out, self.parameter1, self.parameter2))
        return out


class AutoMonadTwoAssignTwoAddnDependencyBenchmarkNet(Cell):
    def __init__(self):
        super().__init__()
        self.parameter2 = ms.Parameter(Tensor([3.0], ms.float32), name="parameter2")
        self.addN = P.AddN()

    def construct(self, inputs):
        out = self.addN((inputs, inputs, self.parameter2))
        out = self.addN((out, inputs, inputs))
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_monad_read_dependency_two_assign_two_addn():
    net = AutoMonadTwoAssignTwoAddnDependencyNet()
    benchmarknet = AutoMonadTwoAssignTwoAddnDependencyBenchmarkNet()
    out1 = net(Tensor([9.0], ms.float32))
    out2 = benchmarknet(Tensor([9.0], ms.float32))
    allclose_nparray(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)
