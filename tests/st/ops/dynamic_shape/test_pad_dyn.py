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
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class PadNet(nn.Cell):
    def __init__(self, paddings):
        super(PadNet, self).__init__()
        self.paddings = paddings

    def construct(self, x):
        return ops.pad(x, self.paddings)


def run_case():
    paddings = ((1, 0), (0, 2))
    paddings_ms = (0, 2, 1, 0)
    shape = (4, 4)
    shape_dyn = (None, 4)
    sz = reduce(lambda a, b: a * b, shape)
    x = np.arange(sz).reshape(shape).astype(np.float32)
    expect = np.pad(x, paddings, mode="constant", constant_values=0)
    x_dyn = Tensor(shape=shape_dyn, dtype=mindspore.float32)
    net = PadNet(paddings_ms)
    net.set_inputs(x_dyn)
    output = net(Tensor(x))
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pad_dyn_cpu():
    """
    Feature: test Pad dynamic shape on CPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_case()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pad_dyn_gpu():
    """
    Feature: test Pad dynamic shape on GPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    run_case()
