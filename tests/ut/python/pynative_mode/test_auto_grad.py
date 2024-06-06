# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" test_auto_grad """

from mindspore.ops import composite as C
from mindspore import Tensor
from mindspore import nn
import mindspore


class MultiInputNet(nn.Cell):
    def __init__(self):
        super(MultiInputNet, self).__init__()

    def construct(self, x, t):
        y = x * x
        y = y * t[0]
        z = self.network(y)
        k = y + z
        return k


def test_auto_grad_case1():
    """
    Feature: Test auto grad case1.
    Description: Test COOTensor GetAttr.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    net = MultiInputNet()
    C.GradOperation()
