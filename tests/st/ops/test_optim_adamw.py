from __future__ import absolute_import
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import torch
import mindspore
from mindspore import nn
from mindspore import Tensor, context
from mindspore.mint.optim import AdamW
from mindspore.experimental.optim.lr_scheduler import StepLR


class Network(nn.Cell):
    def __init__(self, lin_weight, lin_bias):
        super().__init__()
        self.lin = nn.Dense(2, 3, weight_init=lin_weight, bias_init=lin_bias)
        self.relu = nn.ReLU()

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


class AdamWFactory():
    def __init__(self, group=True, lr_dynamic=False, if_change=False, dtype=np.float32, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2, amsgrad=False, maximize=False):
        super().__init__()
        np.random.seed(1024)
        self.lin_weight_np = np.random.randn(3, 2).astype(dtype)
        self.lin_bias_np = np.random.randn(3,).astype(dtype)

        self.data = np.random.rand(2, 2).astype(np.float32)
        self.label = np.random.rand(2, 3).astype(np.float32)

        self.group = group
        self.lr_dynamic = lr_dynamic
        self.if_change = if_change
        self.epochs = 1
        self.steps = 1
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, eps=self.eps, betas=self.betas,
                                          weight_decay=self.weight_decay, amsgrad=self.amsgrad, maximize=self.maximize)
        else:
            bias_params, no_bias_params = [], []
            for param in model.named_parameters():
                if "bias" in param[0]:
                    bias_params.append(param[1])
                else:
                    no_bias_params.append(param[1])
            group_params = [{'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "betas": (0.88, 0.8)},
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
        return output.detach().numpy()


    def forward_mindspore_impl(self):
        lin_weight = Tensor(self.lin_weight_np.copy())
        lin_bias = Tensor(self.lin_bias_np.copy())
        model = Network(lin_weight, lin_bias)

        data = Tensor(self.data)
        label = Tensor(self.label)

        if not self.group:
            optimizer = AdamW(model.trainable_params(), lr=self.lr, eps=self.eps, betas=self.betas,
                              weight_decay=self.weight_decay, amsgrad=self.amsgrad, maximize=self.maximize)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model.trainable_params()))
            no_bias_params = list(filter(lambda x: 'bias' not in x.name, model.trainable_params()))
            group_params = [{'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "betas": (0.88, 0.8)},
                            {'params': no_bias_params, 'lr': 0.66, "amsgrad": True}]
            optimizer = AdamW(params=group_params, lr=self.lr)

        criterion = nn.MAELoss(reduction="mean")
        lr_scheduler = StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)

        def forward_fn(data, label):
            logits = model(data)
            loss = criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamw_basic(mode):
    """
    Feature: Test adamw.
    Description: Test adamw with default parameter.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(False, False)
    fact.result_cmp()
    fact = AdamWFactory(False, False, betas=(0.8, 0.888), weight_decay=1e-3)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamw_basic_amsgrad(mode):
    """
    Feature: Test adamw.
    Description: Test adamw with default parameter.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(False, False, lr=1e-2, betas=(0.8, 0.888), weight_decay=1e-3, amsgrad=True)
    fact.result_cmp()
    fact = AdamWFactory(False, False, amsgrad=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamw_basic_maximize(mode):
    """
    Feature: Test adamw.
    Description: Test adamw with default parameter.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(False, False, maximize=True)
    fact.result_cmp()
    fact = AdamWFactory(False, False, amsgrad=True, maximize=True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamw_group(mode):
    """
    Feature: Test adamw.
    Description: Test adamw with grouped params.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(True, False)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamw_lr_dynamic(mode):
    """
    Feature: Test adamw.
    Description: Test adamw when lr is dynamic.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(False, True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamw_group_lr_dynamic(mode):
    """
    Feature: Test adamw.
    Description: Test adamw with grouped params when lr is dynamic.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(True, True)
    fact.result_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level3', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_adamw_group_lr_dynamic_change_param(mode):
    """
    Feature: Test adamw.
    Description: Test adamw with grouped params when optimizer params are changed.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.LAX)
    fact = AdamWFactory(True, True, True)
    fact.result_cmp()
