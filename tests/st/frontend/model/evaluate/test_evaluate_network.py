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
import math
import numpy as np
import mindspore.ops.operations as P
import torch
from torch import nn
from torch import optim
import torch.utils.data
from mindspore.train import Model
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.nn import Conv2d
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithEvalCell
from tests.st.frontend.dataset.animal import create_animal_no_random_dataset


def evaluate_torch_model(model, dataloader):
    """
    Evaluate a PyTorch model and return loss, top1, top5 accuracy.
    Assumes model output: [B, num_classes]
    Targets: class indices [B]
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # sum for accurate averaging

    with torch.no_grad():
        for images, targets in dataloader:
            batch_size = images.size(0)
            outputs = model(images)

            # Loss
            total_loss += loss_fn(outputs, targets).item()
            total_samples += batch_size

            # Top-k accuracy
            _, top5_pred = torch.topk(outputs, k=5, dim=1)  # [B, 5]
            top5_pred = top5_pred.t()  # [5, B]
            correct = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))

            top1_correct += correct[:1].reshape(-1).sum().item()
            top5_correct += correct[:5].reshape(-1).sum().item()

    avg_loss = total_loss / total_samples
    top1_acc = top1_correct / total_samples
    top5_acc = top5_correct / total_samples

    return {
        'loss': avg_loss,
        'top1': top1_acc,
        'top5': top5_acc
    }


class TorchNet(nn.Module):
    def __init__(self, weight_init):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, bias=False)
        self.conv_weight = weight_init
        self._initialize_weights()

    def _initialize_weights(self):
        self.conv.register_parameter("weight", nn.Parameter(self.conv_weight))

    def forward(self, x):
        out = self.conv(x)
        out = torch.mean(out, (2, 3))
        return out


class TrainMeNet(Cell):
    def __init__(self, weight_init):
        super().__init__()
        self.conv = Conv2d(in_channels=3, out_channels=12, kernel_size=1, weight_init=weight_init)
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

class Factory:
    def __init__(self, epoch_size=1, num_classes=12, batch_size=32):
        super().__init__()
        self.epoch_size = epoch_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.weight_init = np.random.randn(12, 3, 1, 1).astype(np.float32) * 0.01
        self.loss = NetLoss()

    @staticmethod
    def get_data_loader():
        ds = create_animal_no_random_dataset(epoch_size=1)
        for data in ds.create_tuple_iterator(output_numpy=True):
            images = torch.from_numpy(data[0])
            target = torch.from_numpy(np.argmax(data[1], axis=1))

            dataset = torch.utils.data.dataset.TensorDataset(images, target)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                                 pin_memory=True)
        return loader

    def train_conv_reducemean_pt(self, metrics_param=None, iftrain=True, ifeval=True, epochs=1):
        model = TorchNet(torch.from_numpy(self.weight_init.astype(np.float32)))
        loader = self.get_data_loader()

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

    def train_conv_reducemean_me_eval_network(self, metrics_param=None, iftrain=None, ifeval=None,
                                              epochs=None, evalnet=None, trainnet=None,
                                              indexes=None):
        ds_train = create_animal_no_random_dataset(epoch_size=epochs)
        ds_eval = create_animal_no_random_dataset(epoch_size=epochs)

        opt = Momentum(learning_rate=0.1, momentum=0.9, params=trainnet.get_parameters())

        if trainnet is None:
            raise Exception("trainnet is  None!!!")
        model = Model(trainnet, self.loss, opt, eval_network=evalnet, eval_indexes=indexes,
                      metrics=metrics_param)

        if iftrain is True:
            model.train(epochs, ds_train, dataset_sink_mode=True)
        output = None
        if ifeval is True:
            output = model.eval(ds_eval, dataset_sink_mode=True)
        return output


def test_eval_network_net1_net2():
    """
    Feature: test evaluation.
    Description: test evaluation.
    Expectation: the result match with expected result.
    """
    fact = Factory()
    network1 = TrainMeNet(Tensor(fact.weight_init))
    network2 = WithEvalCell(network1, fact.loss)
    metrics_param_me = {"loss", "top_1_accuracy", "top_5_accuracy"}
    metrics_param_pt = {"loss", "top1", "top5"}
    output = fact.train_conv_reducemean_pt(metrics_param=metrics_param_pt, iftrain=False)
    output2 = fact.train_conv_reducemean_me_eval_network(metrics_param_me, evalnet=network2,
                                                         trainnet=network1, iftrain=False,
                                                         ifeval=True, epochs=1, indexes=[0, 1, 2])
    assert math.isclose(output['loss'] + 0.01, output2['loss'] + 0.01, rel_tol=1e-2)
    assert output2['top_1_accuracy'] == output['top1']
    assert output2['top_5_accuracy'] == output['top5']
