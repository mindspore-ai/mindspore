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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit


class Net(nn.Cell):
    def __init__(self, output_size, return_indices):
        super(Net, self).__init__()
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(output_size, return_indices)

    @jit
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adaptive_max_pool2d():
    """
    Feature: Test adaptive_max_pool2d ops.
    Description: Test adaptive_max_pool2d can run on graph and pynative mode.
    Expectation: testcase passed.
    """
    x = np.random.randn(32, 64, 128, 128).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    adaptive_max_pool2d = Net(64, True)
    output1, _ = adaptive_max_pool2d(Tensor(x))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    adaptive_max_pool2d = Net(64, True)
    output2, _ = adaptive_max_pool2d(Tensor(x))
    assert (output1.asnumpy() == output2.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_adaptive_max_pool2d_to_pooling():
    """
    Feature: Test pooling ops.
    Description: Test adaptive_max_pool2d will be replace with pooling kernel.
    Expectation: testcase passed.
    """
    x = np.random.randn(32, 64, 128, 128).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    adaptive_max_pool2d = Net(40, False)
    output1 = adaptive_max_pool2d(Tensor(x))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    adaptive_max_pool2d = Net(40, False)
    output2 = adaptive_max_pool2d(Tensor(x))
    assert (output1.asnumpy() == output2.asnumpy()).all()
