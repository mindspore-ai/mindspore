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

from functools import reduce
import numpy as np
import pytest
import mindspore
import mindspore.nn as nn
from mindspore.ops.operations import _grad_ops as G
from mindspore import Tensor, context


class SliceGradNet(nn.Cell):
    def __init__(self):
        super(SliceGradNet, self).__init__()
        self.op = G.SliceGrad()

    def construct(self, *args):
        return self.op(*args)


def run_case():
    begin = (1, 2, 3)
    size = (2, 3, 4)
    dy_shape = size
    dy = Tensor(np.arange(reduce(lambda i, j: i * j, dy_shape)).reshape(dy_shape).astype(np.float32))
    x = Tensor(np.random.uniform(10, 20, (8, 6, 12)).astype(np.float32))

    net_static = SliceGradNet()
    expect = net_static(dy, x, begin, size)

    dy_dyn = Tensor(shape=[None, None, None], dtype=mindspore.float32)
    x_dyn = Tensor(shape=[None, None, None], dtype=mindspore.float32)
    net = SliceGradNet()
    net.set_inputs(dy_dyn, x_dyn, begin, size)
    output = net(dy, x, begin, size)
    assert np.allclose(expect.asnumpy(), output.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_grad():
    """
    Feature: aicpu SliceGrad
    Description: test SliceGrad on Ascend.
    Expectation: output compares success with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_case()
