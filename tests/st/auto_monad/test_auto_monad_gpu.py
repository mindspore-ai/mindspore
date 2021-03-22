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
# ==============================================================================
import os
import re
import subprocess
import pytest
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.nn import ReLU, BatchNorm2d, Conv2d, ParameterUpdate
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore import amp
from mindspore import context, Tensor
from mindspore.common import ParameterTuple
from mindspore.common.parameter import Parameter
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


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


class GradOfAllInputs(_Grad):
    '''
    get grads of all inputs
    '''

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


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
    greater = np.greater(error, atol + np.abs(data_me)*rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count/total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".\
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol,
                           atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def clear_files():
    os.system("rm verbose_ir_files/*")


def find_files(file, para):
    output = subprocess.check_output(
        ["grep '%s' verbose_ir_files/%s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    return out


class SideEffectCastAll(Cell):
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()
        self.dtype = ms.float16
        np.random.seed(5)
        inputs1 = np.random.randn(5, 5)
        inputs2 = np.random.randn(5, 5)
        self.parameter_a = Parameter(Tensor(inputs1, ms.float32), name="a")
        self.parameter_b = Parameter(Tensor(inputs2, ms.float32), name="b")
        self.assign = P.Assign()

    def construct(self, x, y):
        self.assign(self.parameter_a, x)
        self.assign(self.parameter_b, y)
        out_a = self.cast(self.parameter_a, self.dtype)
        out_b = self.cast(self.parameter_b, self.dtype)
        return out_a, out_b


def test_side_effect_castall():
    clear_files()
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    net = SideEffectCastAll()
    inputs1 = np.random.randn(5, 5)
    inputs2 = np.random.randn(5, 5)
    net(Tensor(inputs1, ms.float32), Tensor(inputs2, ms.float32))
    result = find_files('hwopt*cast_all*.ir', 'CastAll')
    assert result == '2'


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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_side_effect_control_flow_assign_depend_while_net():
    net = SideEffectControlFlowAssignDependWhileNet()
    context.set_context(mode=context.GRAPH_MODE)
    out1 = net(Tensor([9.0], ms.float32), Tensor(
        [99.0], ms.float32), Tensor([1.0], ms.float32))
    net = SideEffectControlFlowAssignDependWhileNet()
    context.set_context(mode=context.PYNATIVE_MODE)
    out2 = net(Tensor([9.0], ms.float32), Tensor(
        [99.0], ms.float32), Tensor([1.0], ms.float32))
    allclose_nparray(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)


class Addn(Cell):
    def __init__(self):
        super().__init__()
        self.parameter3 = Parameter(Tensor([1.0], ms.float32),
                                    name="parameter3")
        self.parameter4 = Parameter(Tensor([3.0], ms.float32),
                                    name="parameter4")
        self.addn = P.AddN()

    def construct(self, inputs):
        out = self.addn((inputs, self.parameter3, self.parameter4))
        return out


class Relu(Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()

    def construct(self, inputs):
        out = self.relu(inputs)
        return out


class SideEffectTwoAssignTwoAddnDependencyNet(Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = Parameter(Tensor([1.0], ms.float32),
                                    name="parameter1")
        self.parameter2 = Parameter(Tensor([3.0], ms.float32),
                                    name="parameter2")
        self.assign = P.Assign()
        self.addN = P.AddN()

    def construct(self, inputs):
        self.assign(self.parameter1, inputs)
        out = self.addN((inputs, self.parameter1, self.parameter2))
        self.assign(self.parameter2, inputs)
        out = self.addN((out, self.parameter1, self.parameter2))
        return out

    def grad_mindspore_impl(self, params, grad_ys):
        grad_net = GradOfAllInputsAndParams(self)
        grad_net.set_train()
        grad_out = grad_net(params, grad_ys)
        return grad_out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ctrl_while_by_while_and_if_in_first_while():
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.sigmoid = P.Sigmoid()
            self.tanh = P.Tanh()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            out = x
            while self.a < 7:
                if self.a < self.c:
                    out = self.relu(x)
                self.a += 1
            while self.c > 5:
                out = self.add(out, out)
                self.c -= 1
            return out

    context.set_context(mode=context.GRAPH_MODE)
    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_a = Tensor(input_np_a)
    net = Net()
    net(input_me_a)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ctrl_while_by_while_and_while_in_first_while():
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.sigmoid = P.Sigmoid()
            self.tanh = P.Tanh()
            self.add = P.Add()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            out = x
            while self.a < self.c:
                out = self.relu(x)
                while self.b > 1:
                    self.b -= 1
                self.a += 1

            while self.c > 5:
                out = self.add(out, out)
                self.c -= 1
            return out

    context.set_context(mode=context.GRAPH_MODE)
    input_np_a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_a = Tensor(input_np_a)
    net = Net()
    net(input_me_a)


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


def test_ir_fusion_inplace_bn_conv_conv():
    clear_files()
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    input_np = np.random.uniform(0.0, 255.0,
                                 size=[4, 4, 4, 4]).astype(np.float32)
    label = np.ones([4, 4, 4, 4]).astype(np.float32)
    net = InplaceNet()
    loss = SoftmaxCrossEntropyWithLogits(sparse=False)
    opt = Momentum(learning_rate=0.01, momentum=0.9,
                   params=filter(lambda x: x.requires_grad, net.get_parameters()))
    net = amp.build_train_network(net, opt, loss, level="O2",
                                  keep_batchnorm_fp32=False)
    net.set_train()
    net(Tensor(input_np), Tensor(label))
    find_accum = find_files("hwopt*cudnn_inplace*ir",
                            "inplace_algo: accumulation")
    find_cover = find_files("hwopt*cudnn_inplace*ir",
                            "inplace_algo: cover")
    assert find_accum == '1'
    assert find_cover == '1'


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


class Add(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x, y):
        return self.add(x, y)


class MixControlNet(Cell):
    def __init__(self, in_channel, x):
        super().__init__()
        #self._save_graphs(save_graph_flag=True, save_graph_path=".")
        self.biasadd = P.BiasAdd()
        self.equal = P.Equal()
        self.addn = P.AddN()
        self.conv = Conv2d(in_channels=in_channel, out_channels=in_channel,
                           kernel_size=1, stride=1, has_bias=False,
                           weight_init='ones', pad_mode='same')
        self.bn = BatchNorm2d(num_features=in_channel)
        self.assignadd = P.AssignAdd()
        self.assign = P.Assign()
        self.relu = ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.bias = Parameter(
            Tensor(np.random.randint(2, size=(3,)).astype((np.float32))),
            name="bias")
        self.bias2 = Parameter(Tensor(np.ones([3]).astype(np.float32)),
                               name="bias2")
        self.parameterupdate = ParameterUpdate(self.bias)
        self.value = Tensor(np.random.randn(*(3,)), ms.float32)
        self.x = x

    def construct(self, input_x):
        x = self.x
        z = self.x
        out = self.biasadd(input_x, self.bias)
        while x < 20:
            update = self.parameterupdate(self.bias2)
            out = self.biasadd(out, update)
            if x < 10:
                out = self.addn((input_x, out))
                while z < 20:
                    out = self.conv(out)
                    z = z + 1
            if x < 20:
                out = self.biasadd(out, self.bias)
                if x % 2 == 0:
                    self.assignadd(self.bias, self.value)
                    out = self.biasadd(out, self.bias)
                    out = self.bn(out)
                else:
                    out = self.conv(out)
            x = x + 1
        out = self.addn((out, out))
        out = self.mean(out, (2, 3))
        return out


def use_build_train_network_controlflow_check_cast_num(network, level, input_x,
                                                       label, cast_num,
                                                       sparse=False,
                                                       loss_flag=True,
                                                       **kwargs):
    opt = Momentum(learning_rate=0.0001, momentum=0.009,
                   params=network.trainable_params())
    loss = None
    if loss_flag:
        loss = SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction='mean')

    train_network = ms.amp.build_train_network(network, opt, loss, level=level,
                                               **kwargs)
    out_me = train_network(input_x, label)
    if context.get_context("mode") == 0:
        content = read_file()
        castnum = re.findall('Cast', content)
        assert len(castnum) == cast_num
    return out_me


def test_auto_mixed_precision_controlflow_auto():
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=True)
    net = MixControlNet(3, 5)
    input_x = Tensor(
        np.random.randint(2, size=(1, 3, 2, 2)).astype((np.float32)))
    label = Tensor(np.zeros([1, 3]).astype(np.float32))
    if ms.context.get_context("device_target") == "Ascend":
        cast_num = 77
    if ms.context.get_context("device_target") == "GPU":
        cast_num = 73
    use_build_train_network_controlflow_check_cast_num(net, "auto", input_x,
                                                       label, cast_num)


def test_updatestate_between_assigns():
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
        updatestate_num = re.findall('UpdateState', content)
        assert len(updatestate_num) == 1


def test_updatestate_between_maketuple_assign():
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
        updatestate_num = re.findall('UpdateState', content)
        assert len(updatestate_num) == 1


def test_updatestate_between_assign_maketuple():
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
        updatestate_num = re.findall('UpdateState', content)
        assert len(updatestate_num) == 1
