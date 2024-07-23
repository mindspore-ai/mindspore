# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore import nn
from mindspore import ops
from mindspore import Tensor, context
import numpy as np


class ArgsPares:
    def __init__(self):
        self.tt1 = 1


class Conv2dMean(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1)
        self.mean = ops.ReduceMean(keep_dims=False)
        self.relu = ops.ReLU()
        self.y = ArgsPares()

    def construct(self, x):
        x = self.relu(x)
        for _ in range(3):
            x = self.y.tt1
        x = self.conv1(x)
        x = self.mean(x, (2, 3))
        return x


def test_catch_exception_of_get_outer_class_attr():
    """
    Feature: Resolve.
    Description: execute this testcase to raise a exception, and print code stack info
        for testcase:test_check_for_body_get_outer_class_attr_log.py::test_catch_exception_stack_trace_log
    Expectation: raise exception with expected code stack info.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.ones((3, 32, 32)).astype(np.float32))
    Conv2dMean()(x)
