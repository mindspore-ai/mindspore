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
import random
import torch
import mindspore
import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore.experimental.optim import SGD
from mindspore.experimental.optim import Adam
from mindspore.experimental.optim import AdamW
from mindspore.graph.api import jit
from mindspore.experimental import optim


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


class CosineAnnealingWarmRestartsFactory:
    def __init__(self, group=0, epoch=1, steps=1, lr=0.1, dtype=np.float32, lr_step=None,
                 lr_epoch=None, t_0=0, t_mult=1, eat_min=0, last_epoch=-1):
        super().__init__()
        self.index = random.randint(1, 3)
        self.group = group
        self.lin_weight_np = np.random.randn(3, 2).astype(dtype)
        self.lin_bias_np = np.random.randn(3, ).astype(dtype)
        self.data = np.random.rand(2, 2).astype(np.float32)
        self.label = np.random.rand(2, 3).astype(np.float32)
        self.epochs = epoch
        self.steps = steps
        self.lr = lr
        self.lr_step = lr_step
        self.lr_epoch = lr_epoch
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eat_min = eat_min
        self.last_epoch = last_epoch

    def forward_pytorch_impl(self):
        lin_weight = torch.Tensor(self.lin_weight_np.copy())
        lin_bias = torch.Tensor(self.lin_bias_np.copy())

        model = NetworkPt()
        model.lin.weight = torch.nn.Parameter(lin_weight)
        model.lin.bias = torch.nn.Parameter(lin_bias)

        data = torch.from_numpy(self.data.copy())
        label = torch.from_numpy(self.label.copy())

        optimizer = None
        if self.group == 0:
            if self.index == 1:
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
            elif self.index == 2:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            elif self.index == 3:
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        else:
            bias_params, no_bias_params = [], []
            for param in model.named_parameters():
                if "bias" in param[0]:
                    bias_params.append(param[1])
                else:
                    no_bias_params.append(param[1])
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "dampening": 1,
                 "initial_lr": 0.9},
                {'params': no_bias_params, 'lr': 0.66, "momentum": 0.7, "nesterov": True,
                 "initial_lr": 0.66}]
            if self.index == 1:
                optimizer = torch.optim.SGD(params=group_params, lr=self.lr)
            elif self.index == 2:
                optimizer = torch.optim.Adam(params=group_params, lr=self.lr)
            elif self.index == 3:
                optimizer = torch.optim.AdamW(params=group_params, lr=self.lr)
        criterion = torch.nn.L1Loss(reduction='mean')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.t_0,
                                                                         self.t_mult, self.eat_min,
                                                                         self.last_epoch)

        for _ in range(self.epochs):
            if self.lr_epoch is not None:
                for _ in range(self.lr_epoch):
                    scheduler.step()

            for _ in range(self.steps):
                if self.steps is not None:
                    for _ in range(self.steps):
                        scheduler.step()

                optimizer.zero_grad()
                loss = criterion(model(data), label)
                loss.backward()
                optimizer.step()

        model(data)
        pt_last_lr = scheduler.get_last_lr()
        return pt_last_lr

    def forward_mindspore_impl(self):
        lin_weight = Tensor(self.lin_weight_np.copy())
        lin_bias = Tensor(self.lin_bias_np.copy())
        model_ms = Network(lin_weight, lin_bias)

        data = Tensor(self.data)
        label = Tensor(self.label)
        optimizer = None
        if self.group == 0:
            if self.index == 1:
                optimizer = SGD(params=model_ms.trainable_params(), lr=self.lr)
            elif self.index == 2:
                optimizer = Adam(params=model_ms.trainable_params(), lr=self.lr)
            elif self.index == 3:
                optimizer = AdamW(params=model_ms.trainable_params(), lr=self.lr)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model_ms.trainable_params()))
            no_bias_params = list(
                filter(lambda x: 'bias' not in x.name, model_ms.trainable_params()))
            group_params = [
                {'params': bias_params, 'weight_decay': 0.01, 'lr': 0.9, "dampening": 1,
                 "initial_lr": Tensor(0.9)},
                {'params': no_bias_params, 'lr': 0.66, "momentum": 0.7, "nesterov": True,
                 "initial_lr": Tensor(0.66)}]
            if self.index == 1:
                optimizer = SGD(params=group_params, lr=self.lr)
            elif self.index == 2:
                optimizer = Adam(params=group_params, lr=self.lr)
            elif self.index == 3:
                optimizer = AdamW(params=group_params, lr=self.lr)

        criterion = nn.MAELoss(reduction="mean")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.t_0,
                                                                   self.t_mult, self.eat_min,
                                                                   self.last_epoch)

        def forward_fn(data, label):
            logits = model_ms(data)
            loss = criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        @jit
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss

        def train(epochs, steps):
            for _ in range(epochs):
                if self.lr_epoch is not None:
                    for _ in range(self.lr_epoch):
                        scheduler.step()
                for _ in range(steps):
                    if self.steps is not None:
                        for _ in range(self.steps):
                            scheduler.step()
                    train_step(data, label)

        train(self.epochs, self.steps)
        ms_last_lr = scheduler.get_last_lr()
        return ms_last_lr

    def result_cmp(self):
        if self.group:
            out_pt1 = self.forward_pytorch_impl()[0]
            out_pt2 = self.forward_pytorch_impl()[1]
            out_pt_float1 = round(out_pt1, 5)
            out_pt_float2 = round(out_pt2, 5)
            out_ms1 = self.forward_mindspore_impl()[0]
            out_ms2 = self.forward_mindspore_impl()[1]
            out_ms_float1 = round(float(out_ms1), 5)
            out_ms_float2 = round(float(out_ms2), 5)
            assert out_pt_float1 == out_ms_float1
            assert out_pt_float2 == out_ms_float2
        else:
            out_pt = self.forward_pytorch_impl()[0]
            out_pt_float = round(out_pt, 5)
            out_ms = self.forward_mindspore_impl()[0]
            out_ms_float = round(float(out_ms), 5)
            assert out_ms_float == out_pt_float



def test_cosine_annealing_warm_restarts_epoch_1_step_0_group_0():
    """
    Feature: test cosine annealing lr
    Description: test cosine annealing lr.
    Expectation: the result match with expected result.
    """
    fact = CosineAnnealingWarmRestartsFactory(epoch=2, group=0, steps=4, lr=0.1, lr_step=0,
                                              lr_epoch=1, t_0=5, t_mult=1, eat_min=0, last_epoch=-1)
    fact.result_cmp()
