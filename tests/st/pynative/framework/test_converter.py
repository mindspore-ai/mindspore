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
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_pynative_pipeline():
    """
    Feature: Pynative input converter
    Description: Test converter
    Expectation: run success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()

        def construct(self, input1, input2):
            x1 = input1.hardshrink()
            x2 = input2.ptp(1)
            output = x1 * self.add(x1, x2)
            return output

    input1 = Tensor(np.random.randn(1, 2, 3, 4, 5, 6, 7), ms.float32, const_arg=True)
    input2 = Tensor(np.random.randn(1, 2, 3, 4, 5, 6, 7), ms.float16, const_arg=True)

    net = Net()
    output = net(input1, input2)
    x1 = input1.hardshrink()
    x2 = input2.ptp(1)
    expect = x1 * (x1 + x2)
    assert output.asnumpy().all() == expect.all()
