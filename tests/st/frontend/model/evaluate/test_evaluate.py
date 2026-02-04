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
"""Test evaluation of network."""
import numpy as np
import math
import torch
from torch import nn
from torch import optim
import torch.utils.data
from mindspore.train import Model
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.nn import Conv2d
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from tests.st.frontend.dataset.animal import create_animal_no_random_dataset
from .test_evaluate_network import evaluate_torch_model


class MeNet(Cell):
    def __init__(self, weight_init, in_channels=3, out_channels=12, kernel_size=1):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, weight_init=weight_init)
        self.reduce = P.ReduceMean()

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.reduce(out, (2, 3))
        return out


class NetLoss(Cell):
    def __init__(self):
        super().__init__()
        self.loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        self.reduce = P.ReduceMean()

    def construct(self, inputs, label):
        out = self.loss(inputs, label)
        out = self.reduce(out, (0,))
        return out


class TorchNet(nn.Module):
    def __init__(self, weight_init, in_channels=3, out_channels=12, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, bias=False)
        self.conv_weight = weight_init
        self._initialize_weights()

    def _initialize_weights(self):
        self.conv.register_parameter("weight", nn.Parameter(self.conv_weight))

    def forward(self, x):
        out = self.conv(x)
        out = torch.mean(out, (2, 3))
        return out


class EvaluteFactory:
    def __init__(self, epoch_size=1, num_classes=12, batch_size=32, weight_shape=(12, 3, 1, 1)):
        super().__init__()
        np.random.seed(5)
        self.epoch_size = epoch_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.weight_init = np.random.randn(*weight_shape).astype(np.float32) * 0.01
        self.loss = NetLoss()

    def get_data_loader(self):
        ds = create_animal_no_random_dataset(epoch_size=1)
        for data in ds.create_tuple_iterator(output_numpy=True):
            images = torch.from_numpy(data[0])
            target = torch.from_numpy(np.argmax(data[1], axis=1))
            dataset = torch.utils.data.dataset.TensorDataset(images, target)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                 pin_memory=True)
        return loader

    def train_conv_reducemean_me(self, metrics_param=None, iftrain=True, ifeval=True, epochs=1):
        net = MeNet(Tensor(self.weight_init))
        ds_train = create_animal_no_random_dataset(epoch_size=epochs)
        ds_eval = create_animal_no_random_dataset(epoch_size=epochs)

        opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())

        model = Model(net, self.loss, opt, metrics=metrics_param)
        if iftrain is True:
            model.train(epochs, ds_train, dataset_sink_mode=True)
        output = None
        if ifeval is True:
            output = model.eval(ds_eval, dataset_sink_mode=True)
        return output

    def train_conv_reducemean_pt(self, metrics_param=None, iftrain=True, ifeval=True, epochs=1):
        model = TorchNet(torch.from_numpy(self.weight_init.astype(np.float32)))
        loader = self.get_data_loader()

        loss = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training (if needed)
        if iftrain:
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
            for _ in range(epochs):
                for images, targets in loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()

        # Evaluation
        if ifeval:
            results = evaluate_torch_model(model, loader)
            return results

        return {}


def test_evaluate_input_3():
    """
    Feature: test evaluation.
    Description: test evaluation.
    Expectation: the result match with expected result.
    """
    fact = EvaluteFactory()
    metrics_param_me = {"top_1_accuracy", "top_5_accuracy", "loss"}
    metrics_param_pt = {"loss", "top1", "top5"}
    output = fact.train_conv_reducemean_me(metrics_param_me, iftrain=False)
    results = fact.train_conv_reducemean_pt(metrics_param_pt, iftrain=False)
    assert len(output) == 3
    assert math.isclose(output['loss'] + 0.01, results['loss'] + 0.01, rel_tol=1e-2)
    assert output['top_1_accuracy'] == results['top1']
    assert output['top_5_accuracy'] == results['top5']
