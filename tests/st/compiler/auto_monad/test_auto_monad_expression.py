# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
import shutil
import pytest
import numpy as np
from mindspore.nn import Cell
from mindspore import context, Tensor, Parameter
import mindspore.ops.operations as P
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore as ms
from mindspore import jit
from mindspore.nn import TrainOneStepCell, Momentum
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_auto_monad_addn_adam():
    """
    Feature: Auto monad feature.
    Description: Verify the optimizer operator adam.
    Expectation: No exception.
    """
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
    context.set_context(mode=context.GRAPH_MODE)


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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_auto_monad_read_dependency_two_assign_two_addn():
    """
    Feature: Auto monad feature.
    Description: Verify the effect operator assign.
    Expectation: No exception.
    """
    net = AutoMonadTwoAssignTwoAddnDependencyNet()
    benchmarknet = AutoMonadTwoAssignTwoAddnDependencyBenchmarkNet()
    out1 = net(Tensor([9.0], ms.float32))
    out2 = benchmarknet(Tensor([9.0], ms.float32))
    allclose_nparray(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)


class ForwardNet(Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.weight = Parameter(Tensor(np.array(0), ms.int32), name="param")

    def construct(self, x):
        out = 0
        i = 0
        while i < 3:
            F.assign(self.weight, i)
            out = x * self.weight + out
            i = i + 1
        return out


class BackwardNet(Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_auto_monad_reorder_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_load_convert_tensormove():
    """
    Feature: Auto monad feature: record the value of load.
    Description: record the value of load.
    Expectation: No exception.
    """

    if ms.context.get_context('mode') == 0:
        # set MS_DEV_SIDE_EFFECT_LOAD_ELIM = 0/1/2
        os.environ['MS_DEV_SIDE_EFFECT_LOAD_ELIM'] = '1'
        save_path = "./test_load_convert_tensormove"
        context.set_context(save_graphs=True, save_graphs_path=save_path)
        x = Tensor(np.array(1), ms.int32)
        graph_forword_net = ForwardNet()
        graph_backword_net = BackwardNet(graph_forword_net)
        output_except = (Tensor(np.array(3), ms.int32),)
        graph_mode_grads = graph_backword_net(x)
        content2 = read_file(save_path)
        tensormove_set = re.findall('= TensorMove', content2)
        context.set_context(save_graphs=False)
        try:
            shutil.rmtree(save_path)
        except FileNotFoundError:
            pass
        assert len(tensormove_set) == 3
        assert np.all(graph_mode_grads == output_except)


class ForwardNet2(Cell):
    def __init__(self):
        super(ForwardNet2, self).__init__()
        self.weight = Parameter(Tensor(np.array(0), ms.int32), name="param")

    def construct(self):
        out = 0
        i = 0
        while i < 3:
            F.assign(self.weight, i)
            out = self.weight + out
            i = i + 1
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_load_convert_tensormove_2():
    """
    Feature: Auto monad feature: record the value of load.
    Description: record the value of load.
    Expectation: No exception.
    """
    if ms.context.get_context('mode') == 0:
        os.environ['MS_DEV_SIDE_EFFECT_LOAD_ELIM'] = '1'
        save_path = "./test_load_convert_tensormove2"
        context.set_context(save_graphs=True, save_graphs_path=save_path)
        graph_forword_net = ForwardNet2()
        forward_res = graph_forword_net()
        assert forward_res == 3
        content = read_file(save_path)
        tensormove_set = re.findall('= TensorMove', content)
        context.set_context(save_graphs=False)
        assert len(tensormove_set) == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_load_eliminate():
    """
    Feature: Auto monad feature: test load eliminate.
    Description: test load eliminate.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.variable = Parameter(Tensor(0, ms.float32), name="global")

        def construct(self, x):
            out = self.variable
            self.assign(self.variable, 0)
            out = x ** 2 + self.variable + out
            self.assign(self.variable, 1)
            out = self.variable + out
            return out

    x = Tensor([2], ms.float32)
    net = Net()
    out = net(x)
    assert out == 5


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parameter_tuple_assign():
    """
    Feature: Auto monad feature.
    Description: Parameter tuple assign.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.param1 = Parameter(Tensor(0), name="param1")
            self.param2 = Parameter(Tensor(0), name="param2")

        def construct(self, x):
            params = (self.param1, self.param2)
            self.assign(params[0], x)
            return params[0], params[1]

    x = Tensor(2)
    net = Net()
    out = net(x)
    assert out[0] == 2
    assert out[1] == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parameter_tuple_assign_addn():
    """
    Feature: Auto monad feature.
    Description: Parameter tuple assign and addn.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.addn = P.AddN()
            self.param1 = Parameter(Tensor(1), name="param1")
            self.param2 = Parameter(Tensor(2), name="param2")

        def construct(self, x):
            params = (self.param1, self.param2)
            res1 = self.addn(params)
            self.assign(params[0], x)
            res2 = self.addn(params)
            self.assign(params[1], x * 2)
            res3 = self.addn(params)
            res4 = params[0] + params[1]
            res = (res1, res2, res3, res4)
            return res

    x = Tensor(3)
    net = Net()
    out = net(x)
    assert out == (3, 5, 9, 9)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parameter_tuple_assign_addn_inner_net():
    """
    Feature: Auto monad feature.
    Description: Parameter tuple assign and addn.
    Expectation: No exception.
    """
    class InnerNet(Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.addn = P.AddN()
            self.param1 = Parameter(Tensor(1), name="param1")
            self.param2 = Parameter(Tensor(2), name="param2")

        def construct(self, x):
            params = (self.param1, self.param2)
            res1 = self.addn(params)
            self.assign(params[0], x)
            res2 = self.addn(params)
            res = (res1, res2, self.param1, self.param2)
            return res

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet()
            self.addn = P.AddN()
            self.assign = P.Assign()

        def construct(self, x, y):
            inner_net_res = self.inner_net(x)
            params = (inner_net_res[2], inner_net_res[3])
            out_res1 = self.addn(params)
            self.assign(inner_net_res[2], y)
            out_res2 = self.addn(params)
            self.assign(inner_net_res[3], 2 * y)
            return out_res1 + out_res2, inner_net_res[2] + inner_net_res[3]

    input_x = Tensor(3)
    input_y = Tensor(5)
    net = Net()
    out = net(input_x, input_y)
    assert out == (12, 15)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_parameter_tuple_assign_addn_inner_net_control_flow():
    """
    Feature: Auto monad feature.
    Description: Parameter tuple assign and addn.
    Expectation: No exception.
    """
    class InnerNet(Cell):
        def __init__(self):
            super().__init__()
            self.param1 = Parameter(Tensor(1), name="param1")
            self.param2 = Parameter(Tensor(2), name="param2")

        def construct(self, x):
            if x > 0:
                return self.param1, self.param2
            return self.param2, self.param1

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet()
            self.addn = P.AddN()
            self.assign = P.Assign()

        def construct(self, x, y):
            inner_params = self.inner_net(x)
            out_res1 = self.addn(inner_params)
            self.assign(inner_params[1], y)
            out_res2 = self.addn(inner_params)
            self.assign(inner_params[0], 2 * y)
            return out_res1 + out_res2, inner_params[0] + inner_params[1]

    input_x = Tensor(3)
    input_y = Tensor(5)
    net = Net()
    out = net(input_x, input_y)
    assert out == (9, 15)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_parameter_value_control_flow_ascend():
    """
    Feature: param.value() feature.
    Description: Test param.value() on Ascend platform.
    Expectation: No exception.
    """
    class InnerNet(Cell):
        def __init__(self):
            super().__init__()
            self.param1 = Parameter(Tensor([1]), name="param1")
            self.param2 = Parameter(Tensor([2]), name="param2")
            self.tensor1 = Tensor([3])
            self.tensor2 = Tensor([4])

        def construct(self, x):
            if x > 0:
                return self.param1.value(), self.tensor1
            return self.param2.value(), self.tensor2

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.inner_net = InnerNet()
            self.addn = P.AddN()

        def construct(self, x, y):
            inner_params = self.inner_net(x)
            out_res = self.addn(inner_params) + y
            return out_res, inner_params[0] + inner_params[1]

    input_x = Tensor([3])
    input_y = Tensor([5])
    net = Net()
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_out = net(input_x, input_y)
    assert pynative_out == (9, 4)
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(input_x, input_y)
    assert graph_out == (9, 4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_control_while_for_if_break_parameter():
    """
    Feature: UpdateState grad in pynative mode.
    Description: UpdateState grad with multi inputs.
    Expectation: No exception.
    """
    class Net30(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.Add()
            add_np = np.full((4, 4, 4), 0.5, dtype=np.float32)
            self.add_weight = Parameter(Tensor(add_np), name="add_weight")

        @jit
        def construct(self, x, y, z):
            out = z
            while x < y:
                if 2 * x < y:
                    out = self.add(out, self.add_weight)
                elif 3 * x < y:
                    out = self.relu(out)
                    x = x + 1
                else:
                    break
                x = x + 1

            out = self.relu(out)
            return out

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net30()
    ms_grad = ops.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    ms_grad(net)(Tensor(2), Tensor(20), Tensor(np.random.rand(4, 4, 4), dtype=ms.float32))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_shape_in_auto_monad():
    """
    Feature: Auto monad feature.
    Description: reshape is not a real operator and needs to use its user as input to updatestate.
    Expectation: No exception.
    """
    class DoubleConcatNet(Cell):
        def __init__(self, div_np, concat_np, mul_np, axis=1):
            super().__init__()
            self.div_weight = Parameter(Tensor(div_np), name="div_weight")
            self.concat_weight = Parameter(Tensor(concat_np), name="concat_weight")
            self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
            self.div = P.Div()
            self.concat1 = P.Concat(axis=axis)
            self.concat2 = P.Concat(axis=axis)
            self.mul = P.Mul()
            self.reshape = P.Reshape()

        def construct(self, inputs, label):
            x = self.div(inputs, self.div_weight)
            x = self.concat1((x, x))
            x = self.concat2((x, self.concat_weight))
            x = self.mul(x, self.mul_weight)
            r = self.reshape(self.mul_weight, (1, 32, 8, 2))
            x = self.mul(x, r)
            return x

    _x = Tensor(np.ones([32, 2, 2]), dtype=ms.float32)
    _b = Tensor(np.ones([32, 100]), dtype=ms.float32)

    def compile_net(net):
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        train_net = TrainOneStepCell(net, optimizer)
        train_net.set_train()
        return train_net(_x, _b)

    div_np_input = np.random.randn(32, 2, 2).astype(np.float32)
    concat_np_input = np.random.randn(32, 4, 2).astype(np.float32)
    mul_np_input = np.random.randn(32, 8, 2).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE)
    graph_net = DoubleConcatNet(div_np_input, concat_np_input, mul_np_input)
    graph_out = compile_net(graph_net)
    graph_div_weight = graph_net.div_weight.value()

    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_net = DoubleConcatNet(div_np_input, concat_np_input, mul_np_input)
    pyantive_out = compile_net(pynative_net)
    pyantive_div_weight = pynative_net.div_weight.value()

    assert (graph_div_weight == pyantive_div_weight).all()
    assert (graph_out == pyantive_out).all()
