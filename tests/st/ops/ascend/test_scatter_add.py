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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter


class TestScatterAddNet(nn.Cell):
    def __init__(self, input_x):
        super(TestScatterAddNet, self).__init__()

        self.scattre_add = ops.ScatterAdd()
        self.input_x = Parameter(input_x, name="input_x")

    def construct(self, indices, updates):
        out = self.scattre_add(self.input_x, indices, updates)
        return out


def scattre_add_forward(nptype):
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(nptype))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(nptype))

    expected = np.array([[1.41000e+02, 1.45000e+02, 1.49000e+02],
                         [2.11000e+02, 2.16000e+02, 2.21000e+02],
                         [2.63000e+02, 2.69000e+02, 2.75000e+02]]).astype(nptype)

    net = TestScatterAddNet(inputx)
    output = net(indices, updates)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scattre_add_forward_float16():
    """
    Feature: test scattre_add forward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_add_forward(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_add_forward(np.float16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scattre_add_forward_float32():
    """
    Feature: test scattre_add forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_add_forward(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_add_forward(np.float32)


def scattre_add_dynamic_indices():
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
    indices_dy = Tensor(shape=(2, None), dtype=mindspore.int32)
    updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
                               [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)

    net = TestScatterAddNet(input_x)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = np.array([[1., 1., 1.], [19., 19., 19.]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scattre_add_dynamic_indices():
    """
    Feature: test scattre_add dynamic shape.
    Description: indices is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_add_dynamic_indices()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_add_dynamic_indices()


def scattre_add_dynamic_updates():
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).astype(np.float16))
    indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
    updates = Tensor(np.ones([2, 2, 3]), mindspore.float16)
    updates_dy = Tensor(shape=(2, None, 3), dtype=mindspore.float16)

    net = TestScatterAddNet(input_x)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = np.array([[1., 1., 1.], [3., 3., 3.]]).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scattre_add_dynamic_updates():
    """
    Feature: test scattre_add dynamic shape.
    Description: updates is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_add_dynamic_updates()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_add_dynamic_updates()
