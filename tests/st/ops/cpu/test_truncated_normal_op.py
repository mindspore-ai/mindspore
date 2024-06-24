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
import pytest
import numpy as np
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops.operations.random_ops import TruncatedNormal
from mindspore import nn


class RandomTruncatedNormal(nn.Cell):
    def __init__(self):
        super(RandomTruncatedNormal, self).__init__()
        self.truncatednormal = TruncatedNormal()

    def construct(self, shape):
        return self.truncatednormal(shape)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_truncatednormal_graph():
    """
    Feature: truncatednormal cpu kernel
    Description: Follow normal distribution, with in 2 standard deviations.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor([2, 2], dtype=mindspore.int32)
    truncatednormal_test = RandomTruncatedNormal()
    expect = np.array([2, 2])

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    output = truncatednormal_test(shape)
    assert (output.shape == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_truncatednormal_pynative():
    """
    Feature: truncatednormal cpu kernel
    Description: Follow normal distribution, with in 2 standard deviations.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor([2, 2], dtype=mindspore.int32)
    truncatednormal_test = RandomTruncatedNormal()
    expect = np.array([2, 2])

    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    output = truncatednormal_test(shape)
    assert (output.shape == expect).all()
