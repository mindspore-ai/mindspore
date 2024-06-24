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

import pytest
import numpy as np
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops.operations.random_ops import NonDeterministicInts
from mindspore import nn


class Net(nn.Cell):
    def __init__(self, dtype):
        super(Net, self).__init__()
        self.ndints = NonDeterministicInts(dtype=dtype)

    def construct(self, shape):
        return self.ndints(shape)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nondeterministicints_graph():
    """
    Feature: nondeterministicints gpu kernel
    Description: Generates some integers that match the given type.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor([2, 2], dtype=mindspore.int32)
    ndints_test = Net(dtype=mindspore.int64)
    expect = np.array([2, 2])

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    output = ndints_test(shape)
    assert (output.shape == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nondeterministicints_pynative():
    """
    Feature: nondeterministicints gpu kernel
    Description: Generates some integers that match the given type.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor([2, 2], dtype=mindspore.int64)
    ndints_test = Net(dtype=mindspore.int32)
    expect = np.array([2, 2])

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    output = ndints_test(shape)
    assert (output.shape == expect).all()
