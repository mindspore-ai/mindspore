# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test dynamic_lr."""
import numpy as np
import torch
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.experimental.optim import NAdam
from mindspore.experimental.optim.lr_scheduler import StepLR
from mindspore.experimental.optim.lr_scheduler import LinearLR
from mindspore.graph.api import jit


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


class OptimFactory:
    def __init__(self, group=False, lr_dynamic=False, lr_change=False,
                 out_change=False, inside_change=False, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=4e-3, epochs=1, steps=1, dtype=np.float32):
        super().__init__()
        self.lin_weight_np = np.random.randn(3, 2).astype(dtype)
        self.lin_bias_np = np.random.randn(3, ).astype(dtype)
        self.group = group
        self.lr_dynamic = lr_dynamic
        self.lr_change = lr_change
        self.out_change = out_change
        self.inside_change = inside_change
        self.data = np.random.rand(2, 2).astype(np.float32)
        self.label = np.random.rand(2, 3).astype(np.float32)
        self.epochs = epochs
        self.steps = steps
        self.lr = lr
        self.loss = 1e-4
        self.betas = betas
        self.momentum_decay = momentum_decay
        self.eps = eps
        self.weight_decay = weight_decay

    def forward_pytorch_impl(self):
        lin_weight = torch.Tensor(self.lin_weight_np.copy())
        lin_bias = torch.Tensor(self.lin_bias_np.copy())

        model = NetworkPt()
        model.lin.weight = torch.nn.Parameter(lin_weight)
        model.lin.bias = torch.nn.Parameter(lin_bias)

        data = torch.from_numpy(self.data.copy())
        label = torch.from_numpy(self.label.copy())

        if not self.group:
            optimizer = torch.optim.NAdam(model.parameters(), lr=self.lr, betas=self.betas,
                                          eps=self.eps, momentum_decay=self.momentum_decay,
                                          weight_decay=self.weight_decay)
        else:
            bias_params, no_bias_params = [], []
            for param in model.named_parameters():
                if "bias" in param[0]:
                    bias_params.append(param[1])
                else:
                    no_bias_params.append(param[1])
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "rho": 0.9},
                {'params': no_bias_params, 'lr': 0.66, "rho": 0.7, "eps": 1e-10}]
            optimizer = torch.optim.NAdam(params=group_params, lr=self.lr, betas=self.betas,
                                          eps=self.eps, momentum_decay=self.momentum_decay,
                                          weight_decay=self.weight_decay)
        criterion = torch.nn.L1Loss(reduction='mean')
        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)
        lr_scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / 3,
                                                          end_factor=1.0, total_iters=6,
                                                          last_epoch=-1)

        for _ in range(self.epochs):
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = criterion(model(data), label)
                loss.backward()
                optimizer.step()
                if self.inside_change:
                    optimizer.param_groups[1]["eps"] = 0.2
            if self.out_change:
                optimizer.param_groups[1]["eps"] = 0.2
            if self.lr_dynamic:
                lr_scheduler1.step()
                lr_scheduler2.step()
            if self.lr_change:
                optimizer.param_groups[1]["lr"] = 0.04
        output = model(data)
        return output.detach().numpy()

    def forward_mindspore_impl(self):
        lin_weight = Tensor(self.lin_weight_np.copy())
        lin_bias = Tensor(self.lin_bias_np.copy())
        model_ms = Network(lin_weight, lin_bias)

        data = Tensor(self.data)
        label = Tensor(self.label)

        if not self.group:
            optimizer = NAdam(params=model_ms.trainable_params(), lr=self.lr, betas=self.betas,
                              eps=self.eps, momentum_decay=self.momentum_decay,
                              weight_decay=self.weight_decay)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model_ms.trainable_params()))
            no_bias_params = list(
                filter(lambda x: 'bias' not in x.name, model_ms.trainable_params()))
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "rho": 0.9},
                {'params': no_bias_params, 'lr': 0.66, "rho": 0.7, "eps": 1e-10}]
            optimizer = NAdam(params=group_params, lr=self.lr, betas=self.betas,
                              eps=self.eps, momentum_decay=self.momentum_decay,
                              weight_decay=self.weight_decay)

        criterion = nn.MAELoss(reduction="mean")

        lr_scheduler1 = StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)
        lr_scheduler2 = LinearLR(optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=6,
                                 last_epoch=-1)

        def forward_fn(data, label):
            logits = model_ms(data)
            loss = criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        @jit
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            if self.inside_change:
                optimizer.param_groups[1]["eps"] = 0.2
            return loss

        def train(epochs, steps):
            for _ in range(epochs):
                for _ in range(steps):
                    train_step(data, label)
                if self.out_change:
                    optimizer.param_groups[1]["eps"] = 0.2
                if self.lr_dynamic:
                    lr_scheduler1.step()
                    lr_scheduler2.step()
                if self.lr_change:
                    ops.assign(group_params[1]["lr"], Tensor(0.04))

        train(self.epochs, self.steps)
        output = model_ms(data)
        return output.asnumpy()

    def result_cmp(self):
        loss_out = self.forward_mindspore_impl()
        if self.out_change:
            self.out_change = False
        loss_expect = self.forward_pytorch_impl()
        np.allclose(loss_expect, loss_out, self.loss, self.loss)


def test_nadam_group_dynamic():
    """
    Feature: test nadam group dynamic.
    Description: test nadam group dynamic.
    Expectation: the result match with expected result.
    """
    fact = OptimFactory(group=True, lr_dynamic=True, lr_change=False, out_change=False,
                        inside_change=False, lr=2, betas=(0.5, 0.5),
                        eps=0.8, momentum_decay=0.9, epochs=2, steps=2,
                        weight_decay=1.5)
    # if mindspore.get_context("device_target") == "Ascend":
    fact.loss = 1e-3
    fact.result_cmp()
