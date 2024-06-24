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
from tests.mark_utils import arg_mark
import numpy as np
import pytest


import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations import _inner_ops as P
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_came_part1 = P.ApplyCamePart1()


    def construct(self, grad, eps):
        return self.apply_came_part1(grad, eps)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net():
    """
    Feature: test apply_came_part1 tensor api.
    Description: test inputs given their dtype.
    Expectation: execute without error.
    """
    apply_came_part1 = Net()
    grad = Tensor(np.ones([1024, 64]), dtype=ms.bfloat16)
    output = apply_came_part1(grad, 1.1)
    print(output[0].float().asnumpy())
