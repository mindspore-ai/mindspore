# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_mean """
import mindspore as ms
from mindspore import nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_mean():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.value = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)

        def construct(self):
            return self.value.mean()

    net = Net()
    net()


def test_mean_axis():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.value = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)

        def construct(self):
            return self.value.mean(axis=1)

    net = Net()
    net()


def test_mean_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            return x.mean()

    x = ms.Tensor([[1, 2, 3], [1, 2, 3]], dtype=ms.float32)
    net = Net()
    net(x)


def test_mean_parameter_axis():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            return x.mean(axis=1)

    x = ms.Tensor([[1, 2, 3], [1, 2, 3]], dtype=ms.float32)
    net = Net()
    net(x)
