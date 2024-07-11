# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import os
import re
import pytest
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.numpy as msnp
from mindspore.nn import Cell
from mindspore.nn import ReLU, BatchNorm2d, Conv2d
from mindspore import context, Tensor
from mindspore.common import ParameterTuple
from mindspore.common.parameter import Parameter
from mindspore.ops.composite import GradOperation
from tests.security_utils import security_off_wrap

context.set_context(mode=context.GRAPH_MODE)


class _Grad(Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        if self.real_inputs_count is None or self.sens_param is False:
            if self.wrt_params:
                return self.grad(self.network, self.params)(*inputs)
            return self.grad(self.network)(*inputs)

        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        if self.wrt_params:
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfAllInputsAndParams(_Grad):
    '''
    get grads of all inputs and params
    '''

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, get_by_list=True, sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)


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
        assert np.allclose(data_expected, data_me, rtol,
                           atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


class SideEffectControlFlowAssignDependWhileNet(Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = Parameter(
            Tensor([199.0], ms.float32), name="parameter1")
        self.assign = P.Assign()
        self.assignadd = P.AssignAdd()
        self.addn = P.AddN()

    def construct(self, x, y, z):
        self.assign(self.parameter1, x)
        while self.parameter1 < y:
            x = self.addn((x, x))
            self.assignadd(self.parameter1, z)
        return x

    def grad_mindspore_impl(self, params1, params2, params3, grad_ys):
        grad_net = GradOfAllInputsAndParams(self)
        grad_net.set_train()
        grad_out = grad_net(params1, params2, params3, grad_ys)
        return grad_out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_side_effect_control_flow_assign_depend_while_net():
    """
    Feature: Auto monad feature.
    Description: Verify assign.
    Expectation: No exception.
    """
    net = SideEffectControlFlowAssignDependWhileNet()
    context.set_context(mode=context.GRAPH_MODE)
    out1 = net(Tensor([9.0], ms.float32), Tensor(
        [99.0], ms.float32), Tensor([1.0], ms.float32))
    net = SideEffectControlFlowAssignDependWhileNet()
    context.set_context(mode=context.PYNATIVE_MODE)
    out2 = net(Tensor([9.0], ms.float32), Tensor(
        [99.0], ms.float32), Tensor([1.0], ms.float32))
    allclose_nparray(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)
    context.set_context(mode=context.GRAPH_MODE)


class InplaceNet(Cell):
    def __init__(self):
        super().__init__()
        self.bn1 = BatchNorm2d(num_features=4, eps=1e-4,
                               momentum=0.9, gamma_init=1, beta_init=0,
                               moving_mean_init=0, moving_var_init=1, data_format="NHWC")
        self.bn2 = BatchNorm2d(num_features=4, eps=1e-4,
                               momentum=0.9, gamma_init=1, beta_init=0,
                               moving_mean_init=0, moving_var_init=1, data_format="NHWC")
        self.add = P.Add()
        self.relu = ReLU()
        self.conv2d1 = Conv2d(in_channels=4, out_channels=4,
                              kernel_size=2, data_format="NHWC")
        self.conv2d2 = Conv2d(in_channels=4, out_channels=4,
                              kernel_size=2, data_format="NHWC")
        self.conv2d3 = Conv2d(in_channels=4, out_channels=4,
                              kernel_size=2, data_format="NHWC")
        self.conv2d4 = Conv2d(in_channels=4, out_channels=4,
                              kernel_size=2, data_format="NHWC")

    def construct(self, input_x):
        tmp_c1 = self.conv2d1(input_x)
        tmp_c2 = self.conv2d2(input_x)
        tmp_x = self.bn1(tmp_c1)
        tmp_y = self.bn2(tmp_c2)
        tmp_w = self.add(tmp_x, tmp_y)
        tmp_w = self.relu(tmp_w)

        tmp_c1 = self.conv2d3(tmp_w)
        tmp_c2 = self.conv2d4(tmp_w)
        output = self.add(tmp_c1, tmp_c2)
        return output


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file():
    filename = find_newest_validateir_file('./')
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files('./')
    return content


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_updatestate_between_assigns():
    """
    Feature: Auto monad feature.
    Description: Verify updatestate eliminate.
    Expectation: No exception.
    """
    class UpdateState_Assigns(Cell):
        def __init__(self):
            super().__init__()
            self.para1 = Parameter(Tensor(1, dtype=ms.int32), name='para1')
            self.para2 = Parameter(Tensor(3, dtype=ms.int32), name='para2')

        def construct(self, value1, value2):
            self.para1 = value1
            self.para2 = value2
            return self.para2

    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    input_x = Tensor(10, dtype=ms.int32)
    input_y = Tensor(30, dtype=ms.int32)
    expect = Tensor(30, dtype=ms.int32)
    net = UpdateState_Assigns()
    out = net(input_x, input_y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())
    if ms.context.get_context('mode') == 0:
        content = read_file()
        updatestate_num = re.findall('= UpdateState', content)
        assert len(updatestate_num) == 1


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_updatestate_between_maketuple_assign():
    """
    Feature: Auto monad feature.
    Description: Verify updatestate eliminate.
    Expectation: No exception.
    """
    class UpdateState_MakeTuple_Assign(Cell):
        def __init__(self):
            super().__init__()
            self.para1 = Parameter(Tensor(1, dtype=ms.int32), name='para1')
            self.para2 = Parameter(Tensor(3, dtype=ms.int32), name='para2')
            self.para3 = Parameter(Tensor(5, dtype=ms.int32), name='para3')

        def construct(self, value1, value2, value3):
            (self.para1, self.para2) = (value1, value2)
            self.para3 = value3
            return self.para3

    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    input_x = Tensor(10, dtype=ms.int32)
    input_y = Tensor(30, dtype=ms.int32)
    input_z = Tensor(50, dtype=ms.int32)
    expect = Tensor(50, dtype=ms.int32)
    net = UpdateState_MakeTuple_Assign()
    out = net(input_x, input_y, input_z)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())
    if ms.context.get_context('mode') == 0:
        content = read_file()
        updatestate_num = re.findall('= UpdateState', content)
        assert len(updatestate_num) == 1


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_updatestate_between_assign_maketuple():
    """
    Feature: Auto monad feature.
    Description: Verify updatestate eliminate.
    Expectation: No exception.
    """
    class UpdateState_Assign_MakeTuple(Cell):
        def __init__(self):
            super().__init__()
            self.para1 = Parameter(Tensor(1, dtype=ms.int32), name='para1')
            self.para2 = Parameter(Tensor(3, dtype=ms.int32), name='para2')
            self.para3 = Parameter(Tensor(5, dtype=ms.int32), name='para3')

        def construct(self, value1, value2, value3):
            self.para1 = value1
            (self.para2, self.para3) = (value2, value3)
            return self.para3

    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    input_x = Tensor(10, dtype=ms.int32)
    input_y = Tensor(30, dtype=ms.int32)
    input_z = Tensor(50, dtype=ms.int32)
    expect = Tensor(50, dtype=ms.int32)
    net = UpdateState_Assign_MakeTuple()
    out = net(input_x, input_y, input_z)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())
    if ms.context.get_context('mode') == 0:
        content = read_file()
        updatestate_num = re.findall('= UpdateState', content)
        assert len(updatestate_num) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cycle_parameter_binding():
    """
    Feature: Auto-monad side-effect finder.
    Description: Auto-monad should work properly when cycle parameter binding existed.
    Expectation: Normal output, no core dump.
    """

    class MyActor(Cell):
        def construct(self, inputs):
            return inputs

    class MyCell(Cell):
        def __init__(self, actor_list):
            super().__init__()
            self.zero = Tensor(0, ms.int32)
            self.actor_list = actor_list

        def construct(self, state):
            duration = self.zero
            while duration < 2:
                for n in msnp.arange(3):
                    samples = (state[n])
                    x = self.actor_list[n](samples)
                    print(x)
                duration += 1
            return duration

    actor_list = [MyActor(), MyActor(), MyActor()]
    net = MyCell(actor_list)
    state = Tensor(np.ones((3, 3)), ms.float32)
    out = net(state)
    np.testing.assert_allclose(out.asnumpy(), 2)
