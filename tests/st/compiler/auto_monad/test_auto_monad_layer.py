# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
"""test auto monad"""
# pylint: disable=W0612
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=R1705
import numpy as np
import mindspore as ms
from mindspore import nn, context, Tensor, ParameterTuple
import mindspore.ops.operations as P
from mindspore.ops.composite import GradOperation
from mindspore.dataset import NumpySlicesDataset
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class AutoEncoderTrainNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss_fun = nn.MSELoss()
        self.net = nn.CellList([nn.Dense(2, 32), nn.Dense(32, 2)])
        self.relu = nn.ReLU()

    def reconstruct_sample(self, x: Tensor):
        for _, layer in enumerate(self.net):
            x = layer(x)
            x = self.relu(x)
        return x

    def construct(self, x: Tensor):
        recon_x = self.reconstruct_sample(x)
        return self.loss_fun(recon_x, x)

    def sample_2d_data(self, n_normals=2000, n_outliers=400):
        z = np.random.randn(n_normals, 2)
        outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2))
        centers = np.array([(2., 0), (-2., 0)])
        sigma = 0.3
        normal_points = sigma * z + centers[np.random.randint(len(centers), size=(n_normals,))]
        return np.vstack((normal_points, outliers))

    def create_synthetic_dataset(self):
        transformed_dataset = self.sample_2d_data()
        for dim in range(transformed_dataset.shape[1]):
            min_val = transformed_dataset[:, dim].min()
            max_val = transformed_dataset[:, dim].max()
            if min_val != max_val:
                transformed_dataset[:, dim] = (transformed_dataset[:, dim] - min_val) / (max_val - min_val)
            elif min_val != 1:
                transformed_dataset[:, dim] = transformed_dataset[:, dim] / min_val
        transformed_dataset = transformed_dataset.astype(np.float32)
        return transformed_dataset


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.75)
def test_auto_monad_layer():
    """
    Feature: Auto monad feature.
    Description: Verify auto monad feature.
    Expectation: No exception.
    """
    ae_with_loss = AutoEncoderTrainNetwork()
    transformed_dataset = ae_with_loss.create_synthetic_dataset()
    dataloader = NumpySlicesDataset(data=(transformed_dataset,), shuffle=True)
    dataloader = dataloader.batch(batch_size=16)
    optim = nn.RMSProp(params=ae_with_loss.trainable_params(), learning_rate=0.002,)
    train_net = nn.TrainOneStepCell(ae_with_loss, optim)
    train_net.set_train()
    gen_samples = {}
    num_epoch = 21
    for epoch in range(num_epoch):
        loss = []
        for _, (batch,) in enumerate(dataloader):
            batch = Tensor(batch, dtype=ms.float32)
            loss_ = train_net(batch)
            loss.append(loss_.asnumpy())
        avg_loss = np.array(loss).mean()
        if epoch % 10 == 0:
            gen_samples[epoch] = ae_with_loss.reconstruct_sample(Tensor(transformed_dataset)).asnumpy()
        print(f"epoch: {epoch}/{num_epoch}, avg loss: {avg_loss}")


class SideEffectTwoAssignTwoAddnDependencyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([2.0], ms.float32),
                                       name="parameter1")
        self.parameter2 = ms.Parameter(Tensor([3.0], ms.float32),
                                       name="parameter2")
        self.assign = P.Assign()
        self.addN = P.AddN()

    def construct(self, inputs):
        p1 = self.assign(self.parameter1, inputs)
        out = self.addN((inputs, self.parameter1, self.parameter2))
        p2 = self.assign(self.parameter2, inputs)
        out = self.addN((out, self.parameter1, self.parameter2))
        return out


class SideEffectTwoAddnTwoAssignNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([5.0], ms.float32),
                                       name="parameter1")
        self.parameter2 = ms.Parameter(Tensor([7.0], ms.float32),
                                       name="parameter2")
        self.assign = P.Assign()
        self.addN = P.AddN()

    def construct(self, inputs):
        p1 = self.assign(self.parameter1, inputs)
        out = self.addN((inputs, self.parameter1, self.parameter2))
        out = self.addN((inputs, self.parameter1))
        p2 = self.assign(self.parameter2, inputs)
        return out


class SideEffectAssignTwoAddnReluNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([9.0], ms.float32),
                                       name="parameter1")
        self.assign = P.Assign()
        self.addN = P.AddN()
        self.relu = P.ReLU()

    def construct(self, inputs):
        p1 = self.assign(self.parameter1, inputs)
        out = self.addN((inputs, inputs, inputs))
        out = self.relu(out)
        out = self.addN((self.parameter1, out))
        return out


class SideEffectTwoAssignAddnNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([11.0], ms.float32),
                                       name="parameter1")
        self.parameter2 = ms.Parameter(Tensor([3.0], ms.float32),
                                       name="parameter2")
        self.assign = P.Assign()
        self.addN = P.AddN()

    def construct(self, inputs):
        p1 = self.assign(self.parameter1, inputs)
        p2 = self.assign(self.parameter2, inputs)
        out = self.addN((inputs, self.parameter1, self.parameter2))
        return out


class TwoAddnDependencyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([1.0], ms.float32),
                                       name="parameter1")
        self.parameter2 = ms.Parameter(Tensor([3.0], ms.float32),
                                       name="parameter2")
        self.addN = P.AddN()

    def construct(self, inputs):
        out = self.addN((inputs, self.parameter1, self.parameter2))
        out = self.addN((out, self.parameter1, self.parameter2))
        return out


class TwoAddnNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([1.0], ms.float32),
                                       name="parameter1")
        self.parameter2 = ms.Parameter(Tensor([3.0], ms.float32),
                                       name="parameter2")
        self.addN = P.AddN()

    def construct(self, inputs):
        out = self.addN((inputs, self.parameter1, self.parameter2))
        out = self.addN((inputs, self.parameter1))
        return out


class TwoAddnReluNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter1 = ms.Parameter(Tensor([1.0], ms.float32),
                                       name="parameter1")
        self.addN = P.AddN()
        self.relu = P.ReLU()

    def construct(self, inputs):
        out = self.addN((inputs, inputs, inputs))
        out = self.relu(out)
        out = self.addN((self.parameter1, out))
        return out


class SideEffectLayerInputFinalNet(nn.Cell):
    def __init__(self, layer1, layer2, layer3, layer4, layer6, layer7, layer8):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.funcs = (
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer6, self.layer7,
            self.layer8)

    def construct(self, i, inputs):
        x = self.funcs[i](inputs)
        return x


class _Grad(nn.Cell):
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
        if self.wrt_params:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network, self.params)(*inputs)
            else:
                real_inputs = inputs[:self.real_inputs_count]
                sense_param_inputs = inputs[self.real_inputs_count:]
                return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        else:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network)(*inputs)
            else:
                real_inputs = inputs[:self.real_inputs_count]
                sense_param_inputs = inputs[self.real_inputs_count:]
                return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfAllInputsAndParams(_Grad):
    """
    get grads of all inputs and params
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, get_by_list=True,
                                            sens_param=sens_param),
                         network=network, wrt_params=True, real_inputs_count=real_inputs_count)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_side_effect_switch_layer_pos_i_nine_layer_func():
    """
    Feature: Auto monad feature.
    Description: Verify auto monad feature.
    Expectation: No exception.
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    func1 = SideEffectTwoAssignTwoAddnDependencyNet()
    func2 = SideEffectTwoAddnTwoAssignNet()
    func3 = SideEffectAssignTwoAddnReluNet()
    func4 = SideEffectTwoAssignAddnNet()
    func6 = TwoAddnDependencyNet()
    func7 = TwoAddnNet()
    func8 = TwoAddnReluNet()
    net = SideEffectLayerInputFinalNet(func1, func2, func3, func4, func6, func7, func8)
    input_x = Tensor(np.array([9.0]).astype(np.float32))
    i = Tensor(0, ms.int32)
    netout = net(i, input_x)
    func1_new = SideEffectTwoAssignTwoAddnDependencyNet()
    goodout = func1_new(input_x)
    assert np.allclose(goodout.asnumpy(), netout.asnumpy(), 0.001, 0.001)

    grad_ys = Tensor([18.0], ms.float32)
    back_net = GradOfAllInputsAndParams(net)
    back_out = back_net(i, input_x, grad_ys)

    context.set_context(mode=context.PYNATIVE_MODE)
    func1 = SideEffectTwoAssignTwoAddnDependencyNet()
    goodout = func1(input_x)
    back_net_good = GradOfAllInputsAndParams(func1)
    back_out_good = back_net_good(input_x, grad_ys)
    assert np.allclose(goodout.asnumpy(), netout.asnumpy(), 0.001, 0.001)
    assert len(back_out[0]) == 2
    assert len(back_out[1]) == 12
    assert np.allclose(back_out[0][1].asnumpy(), back_out_good[0][0].asnumpy(), 0.001, 0.001)
    assert np.allclose(back_out[1][0].asnumpy(), back_out_good[1][0].asnumpy(), 0.001, 0.001)
    assert np.allclose(back_out[1][1].asnumpy(), back_out_good[1][1].asnumpy(), 0.001, 0.001)


class Addn(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter3 = ms.Parameter(Tensor([1.0], ms.float32),
                                       name="parameter3")
        self.parameter4 = ms.Parameter(Tensor([3.0], ms.float32),
                                       name="parameter4")
        self.addn = P.AddN()

    def construct(self, inputs):
        out = self.addn((inputs, self.parameter3, self.parameter4))
        return out


class Relu(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()

    def construct(self, inputs):
        out = self.relu(inputs)
        return out


class SideEffectSequentialAssignAddnReluNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.parameter5 = ms.Parameter(Tensor([13.0], ms.float32),
                                       name="parameter5")
        self.memorycell = SideEffectTwoAssignTwoAddnDependencyNet()
        self.addN = Addn()
        self.relu = Relu()
        self.addn = P.AddN()
        self.sequential = nn.SequentialCell([self.relu, self.addN, self.memorycell])

    def construct(self, x, y):
        x = self.sequential(x)
        y = self.sequential(y)
        out = self.addn((x, y, self.parameter5))
        return out

    def grad_mindspore_impl(self, params1, params2, grad_ys):
        grad_net = GradOfAllInputsAndParams(self)
        grad_net.set_train()
        grad_out = grad_net(params1, params2, grad_ys)
        return grad_out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_side_effect_sequential_assign_addn_relu():
    """
    Feature: Auto monad feature.
    Description: Verify auto monad feature.
    Expectation: No exception.
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    net = SideEffectSequentialAssignAddnReluNet()
    out1 = net(Tensor([9.0], ms.float32), Tensor([17.0], ms.float32))
    net = SideEffectSequentialAssignAddnReluNet()
    context.set_context(mode=context.PYNATIVE_MODE)
    out2 = net(Tensor([9.0], ms.float32), Tensor([17.0], ms.float32))
    assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)
