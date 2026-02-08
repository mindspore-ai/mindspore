# Copyright 2024 Huawei Technologies Co., Ltd
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

from __future__ import absolute_import
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore.mint.optim import SGD
from mindspore.experimental.optim.lr_scheduler import StepLR


class Network(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.lin = ms.nn.Linear(2, 3)
        self.relu = ms.nn.ReLU()

    def construct(self, x):
        out = self.lin(x)
        out = self.relu(out)
        return out


class NetworkPt(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.lin(x)
        out = self.relu(out)
        return out


class SGDFactory:
    def __init__(self, group=True, lr_dynamic=False, dtype=np.float32, lr=1e-3, momentum=9., dampening=0.,
                 weight_decay=1e-2, nesterov=False, maximize=False):
        super().__init__()
        np.random.seed(1024)
        self.lin_weight_np = np.random.randn(3, 2).astype(dtype)
        self.lin_bias_np = np.random.randn(3,).astype(dtype)

        self.data = np.random.rand(2, 2).astype(np.float32)
        self.label = np.random.rand(2, 3).astype(np.float32)

        self.group = group
        self.lr_dynamic = lr_dynamic
        self.epochs = 1
        self.steps = 1
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize

    def forward_pytorch_impl(self):
        lin_weight = torch.Tensor(self.lin_weight_np.copy())
        lin_bias = torch.Tensor(self.lin_bias_np.copy())

        model = NetworkPt()
        model.lin.weight = torch.nn.Parameter(lin_weight)
        model.lin.bias = torch.nn.Parameter(lin_bias)

        data = torch.from_numpy(self.data.copy())
        label = torch.from_numpy(self.label.copy())

        if not self.group:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                        dampening=self.dampening, weight_decay=self.weight_decay,
                                        nesterov=self.nesterov, maximize=self.maximize)
        else:
            bias_params, no_bias_params = [], []
            for param in model.named_parameters():
                if "bias" in param[0]:
                    bias_params.append(param[1])
                else:
                    no_bias_params.append(param[1])
            group_params = [{'params': bias_params, 'lr': 0.9, 'weight_decay': 0.01},
                            {'params': no_bias_params, 'lr': 0.66}]
            optimizer = torch.optim.SGD(params=group_params, lr=self.lr)

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

        output = model(data)
        return output.detach().numpy()

    def forward_mindspore_impl(self):
        model = Network()
        model.lin.weight = ms.Parameter(ms.Tensor(self.lin_weight_np.copy()))
        model.lin.bias = ms.Parameter(ms.Tensor(self.lin_bias_np.copy()))

        data = ms.Tensor(self.data)
        label = ms.Tensor(self.label)

        if not self.group:
            optimizer = SGD(model.trainable_params(), lr=self.lr, momentum=self.momentum, dampening=self.dampening,
                            weight_decay=self.weight_decay, nesterov=self.nesterov, maximize=self.maximize)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model.trainable_params()))
            no_bias_params = list(filter(lambda x: 'bias' not in x.name, model.trainable_params()))
            group_params = [{'params': bias_params, 'lr': 0.9, 'weight_decay': 0.01},
                            {'params': no_bias_params, 'lr': 0.66}]
            optimizer = SGD(params=group_params, lr=self.lr)

        criterion = ms.nn.L1Loss(reduction='mean')
        lr_scheduler = StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)

        def forward_fn(data, label):
            logits = model(data)
            loss = criterion(logits, label)
            return loss, logits

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss

        def train(epochs, steps, lr_dynamic):
            for _ in range(epochs):
                for _ in range(steps):
                    train_step(data, label)
                if lr_dynamic:
                    lr_scheduler.step()

        train(self.epochs, self.steps, self.lr_dynamic)
        output = model(data)
        return output.asnumpy()

    def result_cmp(self):
        loss_expect = self.forward_pytorch_impl()
        loss_out = self.forward_mindspore_impl()
        allclose_nparray(loss_expect, loss_out, 0.005, 0.005)


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
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_sgd_basic(mode):
    """
    Feature: Test SGD.
    Description: Test SGD with default parameter.
    Expectation: success.
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.LAX, jit_config={'jit_level': 'O0'})
    fact = SGDFactory(False, False)
    fact.result_cmp()

    # lr
    fact = SGDFactory(False, False, lr=1e-3)
    fact.result_cmp()

    # momentum
    fact = SGDFactory(False, False, momentum=0.1)
    fact.result_cmp()

    # dampening
    fact = SGDFactory(False, False, dampening=0.2)
    fact.result_cmp()

    # weight_decay
    fact = SGDFactory(False, False, weight_decay=1e-3)
    fact.result_cmp()

    # nesterov
    fact = SGDFactory(False, False, nesterov=True)
    fact.result_cmp()

    # maximize
    fact = SGDFactory(False, False, maximize=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_sgd_group(mode):
    """
    Feature: Test SGD.
    Description: Test SGD with grouped params.
    Expectation: success.
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.LAX, jit_config={'jit_level': 'O0'})
    fact = SGDFactory(True, False)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_sgd_lr_dynamic(mode):
    """
    Feature: Test SGD.
    Description: Test SGD when lr is dynamic.
    Expectation: success.
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.LAX, jit_config={'jit_level': 'O0'})
    fact = SGDFactory(False, True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_sgd_group_lr_dynamic(mode):
    """
    Feature: Test SGD.
    Description: Test SGD with grouped params when lr is dynamic.
    Expectation: success.
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.LAX, jit_config={'jit_level': 'O0'})
    fact = SGDFactory(True, True)
    fact.result_cmp()
