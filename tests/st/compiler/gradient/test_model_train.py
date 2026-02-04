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
"""Test Model.train() with jit"""

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore import nn
from mindspore import ops, jit
from mindspore.common.initializer import initializer
from tests.st.pi_jit.share.modeltrain_utils import create_train_model, GeneratorFakeData
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_train_with_nograd_param():
    """
    Feature: Training with no-grad parameter.
    Description: Test network training with parameter that has requires_grad=False.
    Expectation: Training completes without exception.
    Migrated from: test_parse_issue_scenario_supplement.py::test_train_with_nograd_param
    """

    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = ops.Mul()
            self.add = ops.Add()
            self.param_1 = ms.Parameter(Tensor(np.ones((32, 32)), dtype=ms.float32), name='param_1')
            self.param_2 = ms.Parameter(Tensor(np.ones((32, 32)), dtype=ms.float32), name='param_2')

        @jit
        def construct(self, x):
            y = self.mul(x, self.param_1)
            x.add_(self.param_2)
            z = self.add(x, y)
            return z

    net = Network()
    net.param_2.requires_grad = False
    model = create_train_model(net)
    fake_dataset = GeneratorFakeData(size=256, batch_size=32, image_size=(32,), num_classes=32)
    dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
    model.train(2, dataset)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_train_with_param_in_different_ways():
    """
    Feature: Training with parameter passed in different ways.
    Description: Test network with same parameter passed both as network input and as closure variable.
    Expectation: Training completes without exception.
    Migrated from: test_parse_issue_scenario_supplement.py::test_train_with_param_in_different_ways
    """

    class ParamNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = ms.Parameter(Tensor(2, ms.float32), name="myname")

        @ms.jit
        def func(self, same_param):
            return same_param * self.param

        def construct(self, x):
            return self.func(self.param) * x

    net = ParamNet()
    model = create_train_model(net)
    fake_dataset = GeneratorFakeData(size=256, batch_size=32, image_size=(32,), num_classes=32)
    dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
    model.train(2, dataset)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_train_with_depend_and_add_used_in_cpu():
    """
    Feature: Training with depend and add operations on CPU.
    Description: Test network with depend operation and add operation heterogeneous to CPU.
    Expectation: Training completes without exception.
    Migrated from: test_parse_issue_scenario_supplement.py::test_train_with_depend_and_add_used_in_cpu
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign = ops.Assign()
            self.depend = ops.Depend()
            self.add = ops.Add()
            self.mul = ops.Mul()
            self.param = ms.Parameter(initializer("zeros", [32, 32]), name='test_param')

        @jit
        def construct(self, x):
            y = self.mul(x, self.param)
            out = self.assign(self.param, x)
            out_1 = self.depend(Tensor(1.0), out)
            z = self.add(out_1, 1)
            return y + z

    net = Net()
    net.add.add_prim_attr("primitive_target", "CPU")
    model = create_train_model(net)
    fake_dataset = GeneratorFakeData(size=256, batch_size=32, image_size=(32,), num_classes=32)
    dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
    model.train(2, dataset)
