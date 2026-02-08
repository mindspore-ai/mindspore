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
"""Optimizer Base."""
import numpy as np
import torch
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.experimental.optim import SGD
from mindspore.experimental.optim import Adam
from mindspore.experimental.optim import AdamW
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
    def __init__(self, optim_ex="SDG", group=True, lr_dynamic=0, if_change=0, if_change_inside=0,
                 dtype=np.float32):
        super().__init__()
        self.lin_weight_np = np.random.randn(3, 2).astype(dtype)
        self.lin_bias_np = np.random.randn(3, ).astype(dtype)
        self.if_change_inside = if_change_inside
        self.optim_ex = optim_ex
        self.group = group
        self.lr_dynamic = lr_dynamic
        self.if_change = if_change
        self.data = np.random.rand(2, 2).astype(np.float32)
        self.label = np.random.rand(2, 3).astype(np.float32)
        self.epochs = 1
        self.steps = 1
        self.lr = 0.002
        self.loss = 2e-4

    def forward_pytorch_impl(self):
        lin_weight = torch.Tensor(self.lin_weight_np.copy())
        lin_bias = torch.Tensor(self.lin_bias_np.copy())

        model = NetworkPt()
        model.lin.weight = torch.nn.Parameter(lin_weight)
        model.lin.bias = torch.nn.Parameter(lin_bias)

        data = torch.from_numpy(self.data.copy())
        label = torch.from_numpy(self.label.copy())

        optimizer = None
        if not self.group:
            if self.optim_ex == "SDG":
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
            elif self.optim_ex == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            elif self.optim_ex == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        else:
            bias_params, no_bias_params = [], []
            for param in model.named_parameters():
                if "bias" in param[0]:
                    bias_params.append(param[1])
                else:
                    no_bias_params.append(param[1])
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "dampening": 1},
                {'params': no_bias_params, 'lr': 0.66, "momentum": 0.7, "nesterov": True}]
            if self.optim_ex == "SDG":
                optimizer = torch.optim.SGD(params=group_params, lr=self.lr)
            elif self.optim_ex == "Adam":
                optimizer = torch.optim.Adam(params=group_params, lr=self.lr)
            elif self.optim_ex == "AdamW":
                optimizer = torch.optim.AdamW(params=group_params, lr=self.lr)
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
                if self.if_change_inside == 1:
                    optimizer.param_groups[1]["lr"] = 0.04
                elif self.if_change_inside == 2:
                    optimizer.param_groups[1]["nesterov"] = False
                    optimizer.param_groups[1]["momentum"] = 0.2
                    optimizer.param_groups[1]["eps"] = 0.05
            if self.if_change == 1:
                optimizer.param_groups[1]["lr"] = 0.04
            elif self.if_change == 2:
                optimizer.param_groups[1]["nesterov"] = False
                optimizer.param_groups[1]["momentum"] = 0.2
                optimizer.param_groups[1]["eps"] = 0.05
            if self.lr_dynamic == 1:
                lr_scheduler1.step()
            elif self.lr_dynamic == 2:
                lr_scheduler1.step()
                lr_scheduler2.step()

        output = model(data)
        return output.detach().numpy()

    def forward_mindspore_impl(self):
        lin_weight = Tensor(self.lin_weight_np.copy())
        lin_bias = Tensor(self.lin_bias_np.copy())
        model_ms = Network(lin_weight, lin_bias)

        data = Tensor(self.data)
        label = Tensor(self.label)

        optimizer = None
        if not self.group:
            if self.optim_ex == "SDG":
                optimizer = SGD(params=model_ms.trainable_params(), lr=self.lr)
            elif self.optim_ex == "Adam":
                optimizer = Adam(params=model_ms.trainable_params(), lr=self.lr)
            elif self.optim_ex == "AdamW":
                optimizer = AdamW(params=model_ms.trainable_params(), lr=self.lr)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model_ms.trainable_params()))
            no_bias_params = list(
                filter(lambda x: 'bias' not in x.name, model_ms.trainable_params()))
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "dampening": 1},
                {'params': no_bias_params, 'lr': 0.66, "momentum": 0.7, "nesterov": True}]
            if self.optim_ex == "SDG":
                optimizer = SGD(params=group_params, lr=self.lr)
            elif self.optim_ex == "Adam":
                optimizer = Adam(params=group_params, lr=self.lr)
            elif self.optim_ex == "AdamW":
                optimizer = AdamW(params=group_params, lr=self.lr)

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
        def train_step(data, label, if_change_inside):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            if if_change_inside == 1:
                ops.assign(group_params[1]["lr"], Tensor(0.04))
            elif if_change_inside == 2:
                optimizer.param_groups[1]["nesterov"] = False
                optimizer.param_groups[1]["momentum"] = 0.2
                optimizer.param_groups[1]["eps"] = 0.05
            return loss

        def train(epochs, steps, lr_dynamic, if_change):
            for _ in range(epochs):
                for _ in range(steps):
                    train_step(data, label, self.if_change_inside)
                if if_change == 1:
                    ops.assign(group_params[1]["lr"], Tensor(0.04))
                elif if_change == 2:
                    optimizer.param_groups[1]["nesterov"] = False
                    optimizer.param_groups[1]["momentum"] = 0.2
                    optimizer.param_groups[1]["eps"] = 0.05
                if lr_dynamic == 1:
                    lr_scheduler1.step()
                elif lr_dynamic == 2:
                    lr_scheduler1.step()
                    lr_scheduler2.step()

        train(self.epochs, self.steps, self.lr_dynamic, self.if_change)
        output = model_ms(data)
        return output.asnumpy()

    def result_cmp(self):
        loss_out = self.forward_mindspore_impl()
        if self.if_change == 2:
            self.if_change = 0
        loss_expect = self.forward_pytorch_impl()
        np.allclose(loss_expect, loss_out, self.loss, self.loss)
