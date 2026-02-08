# Copyright 2026 Huawei Technologies Co., Ltd
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
"""
Test FusedAdamW operator and network.

This module contains tests for FusedAdamW optimizer using both single operator
and network approaches. Single operator validation uses standard loss criteria,
while network validation allows relaxed precision requirements.
"""
from __future__ import absolute_import

import pytest
import numpy as np
import torch

import mindspore
from mindspore import nn, Tensor, context, mint, Parameter, jit
from mindspore.common.api import _pynative_executor, _cell_graph_executor
from mindspore.common import dtype as mstype
from mindspore.common.dtype import _dtype_to_nptype
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops import operations as P
from mindspore.mint.optim.fused_adamw import FusedAdamW
from mindspore.experimental.optim.lr_scheduler import StepLR

from tests.mark_utils import arg_mark
from tests.st.ops.test_optim_adamw import AdamWFactory as AdamWFactoryBase
from tests.st.ops.test_optim_adamw import allclose_nparray

def get_discontinuous_tensor(low=0, high=1, shape=(4, 9), dim=(1, 0), dtype=mstype.float32):
    x = Tensor(np.random.uniform(low, high, size=shape), dtype)
    output = mint.permute(x, dim)
    return output


class Net(nn.Cell):
    """ Net definition """
    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, net_x):
        net_x = self.biasAdd(self.matmul(net_x, self.weight), self.bias)
        return net_x


class NetWithoutWeight(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    def construct(self, mm_net_x):
        mm_net_x = self.matmul(mm_net_x, mm_net_x)
        return mm_net_x


class Network(nn.Cell):
    def __init__(self, lin_weight, lin_bias, in_channels=2, out_channels=3):
        super().__init__()
        self.lin = nn.Dense(in_channels, out_channels, weight_init=lin_weight, bias_init=lin_bias)
        self.relu = nn.ReLU()

    def construct(self, ms_net_x):
        ms_out = self.lin(ms_net_x)
        ms_out = self.relu(ms_out)
        return ms_out


class NetworkPt(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, pt_net_x):
        pt_out = self.lin(pt_net_x)
        pt_out = self.relu(pt_out)
        return pt_out


class AdamWFactory(AdamWFactoryBase):
    def __init__(self, data_shape=(2, 2), label_shape=(2, 3), weight_shape=(3, 2), bias_shape=(3,),
                 group=True, lr_dynamic=False, if_change=False, dtype=np.float32, data_dtype=np.float32,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 maximize=False, bf16=False, seed_int=3219826299, flag_loss=None):
        super().__init__(group=group, lr_dynamic=lr_dynamic, if_change=if_change,
                        dtype=dtype, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)

        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.dtype = dtype
        self.data_dtype0 = data_dtype
        self.seed_int = seed_int
        self.maximize = maximize

        np.random.seed(self.seed_int)

        self.lin_weight_np = np.random.randn(*weight_shape).astype(dtype)
        self.lin_bias_np = np.random.randn(*bias_shape).astype(dtype)

        self.data = np.random.rand(*data_shape).astype(data_dtype)
        self.label = np.random.rand(*label_shape).astype(data_dtype)
        self.data = (self.data - self.data.mean()) / (self.data.std() + 1e-8)
        self.label = (self.label - self.label.mean()) / (self.label.std() + 1e-8)

        self.bf16 = bf16
        self.loss = 0.001
        if flag_loss is not None:
            self.loss = flag_loss


    def forward_pytorch_impl(self):
        lin_weight = torch.Tensor(self.lin_weight_np.copy().astype(np.float32))
        lin_bias = torch.Tensor(self.lin_bias_np.copy().astype(np.float32))

        model = NetworkPt(self.label_shape[0], self.label_shape[1])
        model.lin.weight = torch.nn.Parameter(lin_weight)
        model.lin.bias = torch.nn.Parameter(lin_bias)

        data = torch.from_numpy(self.data.copy().astype(np.float32))
        label = torch.from_numpy(self.label.copy().astype(np.float32))

        if not self.group:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, eps=self.eps,
                                          betas=self.betas, weight_decay=self.weight_decay,
                                          amsgrad=self.amsgrad, maximize=self.maximize)
        else:
            bias_params, no_bias_params = [], []
            for param in model.named_parameters():
                if "bias" in param[0]:
                    bias_params.append(param[1])
                else:
                    no_bias_params.append(param[1])
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "betas": (0.88, 0.8)},
                {'params': no_bias_params, 'lr': 0.66, "amsgrad": True}]
            optimizer = torch.optim.AdamW(params=group_params, lr=self.lr)

        criterion = torch.nn.L1Loss(reduction='mean')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)

        for _ in range(self.epochs):
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = criterion(model(data), label)
                loss.backward()
                optimizer.step()
            if self.lr_dynamic:
                lr_scheduler.step()
            if self.if_change:
                optimizer.param_groups[1]["betas"] = (0.77, 0.7)
                optimizer.param_groups[1]["amsgrad"] = False

        output = model(data)

        list_params = []
        for parameter in model.parameters():
            list_params.append(parameter.detach().numpy())
        return list_params, output.detach().numpy().astype(self.data_dtype0)

    def forward_mindspore_impl(self):
        if self.bf16:
            data = Tensor(self.data, mindspore.bfloat16)
            self.data = data.float().asnumpy()
            lin_weight = Tensor(self.lin_weight_np.copy(), mindspore.bfloat16)
            self.lin_weight_np = lin_weight.float().asnumpy()
            lin_bias = Tensor(self.lin_bias_np.copy(), mindspore.bfloat16)
            self.lin_bias_np = lin_bias.float().asnumpy()
        else:
            data = Tensor(self.data)
            lin_weight = Tensor(self.lin_weight_np.copy())
            lin_bias = Tensor(self.lin_bias_np.copy())

        label = Tensor(self.label)
        model = Network(lin_weight, lin_bias, self.label_shape[0], self.label_shape[1])

        if not self.group:
            optimizer = FusedAdamW(model.trainable_params(), lr=self.lr, eps=self.eps, betas=self.betas,
                              weight_decay=self.weight_decay, amsgrad=self.amsgrad,
                              maximize=self.maximize)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model.trainable_params()))
            no_bias_params = list(filter(lambda x: 'bias' not in x.name, model.trainable_params()))
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "betas": (0.88, 0.8)},
                {'params': no_bias_params, 'lr': 0.66, "amsgrad": True}]
            optimizer = FusedAdamW(params=group_params, lr=self.lr)

        criterion = nn.MAELoss(reduction="mean")
        lr_scheduler = StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)

        def forward_fn(data, label):
            logits = model(data)
            loss = criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        @jit
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss

        def train(epochs, steps, lr_dynamic, if_change):
            for _ in range(epochs):
                for _ in range(steps):
                    train_step(data, label)
                if lr_dynamic:
                    lr_scheduler.step()
                if if_change:
                    optimizer.param_groups[1]["betas"] = (0.77, 0.7)
                    optimizer.param_groups[1]["amsgrad"] = False

        train(self.epochs, self.steps, self.lr_dynamic, self.if_change)
        output = model(data)

        if data.dtype == mindspore.bfloat16:
            assert output.dtype == mindspore.bfloat16
            output_res = output.float().asnumpy()
        else:
            output_res = output.asnumpy()

        list_params = []
        for parameter in model.trainable_params():
            if data.dtype == mindspore.bfloat16:
                list_params.append(parameter.float().asnumpy())
            else:
                list_params.append(parameter.asnumpy())
        return list_params, output_res

    def result_cmp(self):
        parameters_expect, loss_expect_value = self.forward_pytorch_impl()
        parameters_out, loss_out_value = self.forward_mindspore_impl()

        allclose_nparray(loss_expect_value, loss_out_value, self.loss, self.loss)
        for i, param_out in enumerate(parameters_out):
            allclose_nparray(parameters_expect[i], param_out, self.loss, self.loss)


class AdamWMock(AdamWFactory):
    """Create a basic AdamWFactory with default shapes."""
    def __init__(self, attributes=None, inputs=None):
        data = inputs[0]
        label = inputs[1]
        weight = inputs[2]
        bias = inputs[3]
        betas = attributes.get('betas')
        weight_decay = attributes.get('weight_decay')
        seed_int = attributes.get('seed_int', 3219826299)
        lr = attributes.get('lr', 1e-3)
        amsgrad = attributes.get('amsgrad', False)
        eps = attributes.get('eps', 1e-8)

        super().__init__(data.shape, label.shape, weight.shape, bias.shape, group=True,
                         lr_dynamic=False, if_change=False,
                         dtype=_dtype_to_nptype(weight.dtype),
                         data_dtype=_dtype_to_nptype(data.dtype), lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay,
                         amsgrad=amsgrad, maximize=False, bf16=False, seed_int=seed_int,
                         flag_loss=None)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_graph_compile():
    """
    Feature: Test FusedAdamW.
    Description: Compile training graph in GRAPH_MODE with basic parameters in Graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = FusedAdamW(net.trainable_params(), lr=0.1, weight_decay=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_group1_graph_compile():
    """
    Feature: Test FusedAdamW.
    Description: Assign different lr and weight_decay to two parameter groups in Graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    group_params = [{'params': [all_params[0]], 'lr': 0.0001, 'weight_decay': 0.9},
                    {'params': [all_params[1]]}]
    optimizer = FusedAdamW(group_params, lr=0.1)

    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_group2_graph_compile():
    """
    Feature: Test FusedAdamW.
    Description: Set per-group lr while leaving global lr default in Graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    group_params = [{'params': [all_params[0]], 'lr': 0.02, 'weight_decay': 0.9},
                    {'params': [all_params[1]]}]
    optimizer = FusedAdamW(group_params)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_pynative():
    """
    Feature: Test FusedAdamW.
    Description: Execute one training step in Pynative mode.
    Expectation: Success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = FusedAdamW(net.trainable_params(), lr=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    train_network(inputs, label)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_basic(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with default parameters and custom betas/weight_decay.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=False, lr_dynamic=False)
    fact.result_cmp()
    fact = AdamWFactory(group=False, lr_dynamic=False, betas=(0.8, 0.888), weight_decay=1e-3)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_basic_amsgrad(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with AMSGrad enabled.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=False, lr_dynamic=False, lr=1e-2, betas=(0.8, 0.888), weight_decay=1e-3, amsgrad=True)
    fact.result_cmp()
    fact = AdamWFactory(group=False, lr_dynamic=False, amsgrad=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_basic_maximize(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with maximize and AMSGrad enabled.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=False, lr_dynamic=False, maximize=True)
    fact.result_cmp()
    fact = AdamWFactory(group=False, lr_dynamic=False, amsgrad=True, maximize=True)
    fact.result_cmp()



@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_basic_group(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with grouped params.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=True, lr_dynamic=False)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_basic_lr_dynamic(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with dynamic learning rate scheduler.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=False, lr_dynamic=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_group_lr_dynamic(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with grouped params and dynamic lr.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=True, lr_dynamic=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_group_lr_dynamic_change_param(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with grouped params when optimizer params are changed.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    fact = AdamWFactory(group=True, lr_dynamic=True, if_change=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_base():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with default dtype and extreme beta values (0.0, 1.0).
    Expectation: Success.
    """
    mindspore.set_context(mode=context.PYNATIVE_MODE)
    data_shape = (2, 2)
    label_shape = (2, 3)
    weight_shape = (3, 2)
    bias_shape = (3,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, betas=(0.0, 1.0),
                        weight_decay=1e-3)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mint_optim_fused_adamw_dtype_1000_3000(mode):
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with large matrix dimensions and strict loss tolerance.
    Expectation: Success.
    """
    mindspore.set_context(mode=mode)
    data_shape = (1000, 1000)
    label_shape = (1000, 3000)
    weight_shape = (3000, 1000)
    bias_shape = (3000,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=False,
                        lr_dynamic=False, if_change=False,
                        dtype=np.float32, data_dtype=np.float32, lr=1e-3, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=1e-2, amsgrad=False, maximize=False,
                        flag_loss=0.0002)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_float32():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with with 3D tensor shapes and grouped parameters.
    Expectation: Success.
    """
    data_shape = (25, 25, 25)
    label_shape = (25, 25, 25)
    weight_shape = (25, 25)
    bias_shape = (25,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=True,
                        lr_dynamic=False, if_change=False,
                        dtype=np.float32, data_dtype=np.float32, lr=1e-3, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=1e-2, amsgrad=False, maximize=False)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_20_30():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with grouped params, dynamic lr, and AMSGrad.
    Expectation: Success.
    """
    data_shape = (20, 20)
    label_shape = (20, 30)
    weight_shape = (30, 20)
    bias_shape = (30,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=True,
                        lr_dynamic=True, if_change=True,
                        dtype=np.float32, data_dtype=np.float32, lr=1e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=1e-2, amsgrad=True, maximize=False)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_200_300():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with maximize and custom parameters.
    Expectation: Success.
    """
    data_shape = (200, 200)
    label_shape = (200, 300)
    weight_shape = (300, 200)
    bias_shape = (300,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=False,
                        lr_dynamic=True, if_change=False,
                        dtype=np.float32, data_dtype=np.float32, lr=1e-7, betas=(0.9, 0.999),
                        eps=1e-5, weight_decay=0.1, amsgrad=False, maximize=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_bfloat16():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with bfloat16 precision and AMSGrad.
    Expectation: Success.
    """
    mindspore.set_context(mode=context.PYNATIVE_MODE)
    data_shape = (10, 10)
    label_shape = (10, 10)
    weight_shape = (10, 10)
    bias_shape = (10,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=False, lr=1e-4,
                        betas=(0.8, 0.888),
                        weight_decay=1e-2, amsgrad=True, bf16=True)
    fact.loss = 0.067
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_float32_4d():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with 4D tensor shapes.
    Expectation: Success.
    """
    data_shape = (58, 58, 58, 58)
    label_shape = (58, 58, 58, 58)
    weight_shape = (58, 58)
    bias_shape = (58,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, dtype=np.float32,
                        data_dtype=np.float32)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_float32_5d():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with 5D tensor shapes.
    Expectation: Success.
    """
    data_shape = (9, 9, 9, 9, 9)
    label_shape = (9, 9, 9, 9, 9)
    weight_shape = (9, 9)
    bias_shape = (9,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, dtype=np.float32,
                        data_dtype=np.float32)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_float32_6d():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with 6D tensor shapes and zero learning rate.
    Expectation: Success.
    """
    data_shape = (6, 6, 6, 6, 6, 6)
    label_shape = (6, 6, 6, 6, 6, 6)
    weight_shape = (6, 6)
    bias_shape = (6,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, dtype=np.float32,
                        data_dtype=np.float32, lr=0.0)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_float32_7d():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with 7D tensor shapes.
    Expectation: Success.
    """
    data_shape = (3, 3, 3, 3, 3, 3, 3)
    label_shape = (3, 3, 3, 3, 3, 3, 3)
    weight_shape = (3, 3)
    bias_shape = (3,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, dtype=np.float32,
                        data_dtype=np.float32)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_float32_8d():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with 8D tensor shapes.
    Expectation: Success.
    """
    data_shape = (6, 6, 6, 6, 6, 6, 6, 6)
    label_shape = (6, 6, 6, 6, 6, 6, 6, 6)
    weight_shape = (6, 6)
    bias_shape = (6,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, dtype=np.float32,
                        data_dtype=np.float32)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_base_discontinuous_tensor():
    """
    Feature: Test FusedAdamW with non-contiguous tensors
    Description: Test FusedAdamW using non-contiguous tensors.
    Expectation: Success.
    """
    data_shape = (2, 2)
    label_shape = (3, 2)
    weight_shape = (2, 3)
    bias_shape = (3,)

    data = get_discontinuous_tensor(shape=data_shape)
    label = get_discontinuous_tensor(shape=label_shape)
    weight = get_discontinuous_tensor(shape=weight_shape)
    bias_data = Tensor(np.random.randn(1).astype(np.float32))
    bias = mint.broadcast_to(bias_data, bias_shape)

    fact = AdamWMock(inputs=[data, label, weight, bias],
                     attributes={'betas': (0.8, 0.888), 'weight_decay': 1e-3})
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_float16_group_true_5d():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with float16 precision, 5D tensors, grouped params, AMSGrad and maximize.
    Expectation: Success.
    """
    data_shape = (8, 8, 8, 8, 8)
    label_shape = (8, 8, 8, 8, 8)
    weight_shape = (8, 8)
    bias_shape = (8,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=True,
                        lr_dynamic=False, if_change=False,
                        dtype=np.float16, data_dtype=np.float16, lr=0.1, betas=(0.9, 0.999),
                        eps=1e-7, weight_decay=1e-3, amsgrad=True, maximize=True, flag_loss=0.002)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_float64():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with float64 TypeError.
    Expectation: Raise TypeError due to unsupported float64 dtype.
    """
    data_shape = (2, 2)
    label_shape = (2, 3)
    weight_shape = (3, 2)
    bias_shape = (3,)

    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, dtype=np.float64,
                        betas=(0.8, 0.888), weight_decay=1e-3)
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_withoutparam():
    """
    Feature: Test FusedAdamW.
    Description: Test constructor with empty parameter list.
    Expectation: Raise ValueError indicating params must not be empty.
    """

    net = NetWithoutWeight()
    net.set_train()
    with pytest.raises(ValueError, match=r"For 'Optimizer', the argument params must not be empty."):
        FusedAdamW(net.trainable_params(), lr=0.1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_lr_neg():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with invalid learning_rate.
    Expectation: Raise ValueError due to invalid learning rate.
    """
    data_shape = (2, 2)
    label_shape = (2, 3)
    weight_shape = (3, 2)
    bias_shape = (3,)
    learning_rates = -0.1
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, lr=learning_rates)
    with pytest.raises(ValueError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_20_30_lr_int():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with integer learning rate instead of float.
    Expectation: Raise TypeError due to invalid learning rate type.
    """
    data_shape = (20, 20)
    label_shape = (20, 30)
    weight_shape = (30, 20)
    bias_shape = (30,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=True,
                        lr_dynamic=True, if_change=True,
                        dtype=np.float32, data_dtype=np.float32, lr=1, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=1e-2, amsgrad=True, maximize=False)
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_betas_neg():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with negative beta values.
    Expectation: Raise ValueError due to invalid beta parameters.
    """
    data_shape = (2, 2)
    label_shape = (2, 3)
    weight_shape = (3, 2)
    bias_shape = (3,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=False,
                        lr_dynamic=True, if_change=False,
                        dtype=np.float32, data_dtype=np.float32, lr=1e-7, betas=(-0.9, -0.1),
                        eps=1e-5, weight_decay=0.1, amsgrad=False, maximize=True)
    with pytest.raises(ValueError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_mint_optim_fused_adamw_dtype_weight_neg():
    """
    Feature: Test FusedAdamW
    Description: Test FusedAdamW with negative weight decay value.
    Expectation: Raise ValueError due to invalid weight decay.
    """
    data_shape = (2, 2)
    label_shape = (2, 3)
    weight_shape = (3, 2)
    bias_shape = (3,)
    fact = AdamWFactory(data_shape, label_shape, weight_shape, bias_shape, group=False,
                        lr_dynamic=True, if_change=False,
                        dtype=np.float32, data_dtype=np.float32, lr=1e-7, betas=(0.9, 0.999),
                        eps=1e-5, weight_decay=-0.1, amsgrad=False, maximize=True)
    with pytest.raises(ValueError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()
