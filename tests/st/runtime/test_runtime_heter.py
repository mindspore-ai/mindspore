# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.batch_size = 32
        self.weight1 = Parameter(Tensor(np.ones([6, 1, 5, 5]).astype(np.float16)), name="weight")
        self.weight2 = Parameter(Tensor(np.ones([16, 6, 5, 5]).astype(np.float16)), name="weight")

        self.relu = P.ReLU()
        self.relu_cpu = P.ReLU()
        self.relu_cpu.set_device("CPU")
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init=self.weight1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init=self.weight2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu_cpu(output)
        return output


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_lenet():
    """
    Feature: Runtime special format in the heterogeneous scene.
    Description: Test special format in the heterogeneous scene.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float16) * 0.01)
    net = LeNet()
    net(data)
