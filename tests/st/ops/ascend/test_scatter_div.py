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

import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter

# all cases tested against dchip


class TestScatterDivNet(nn.Cell):
    def __init__(self, inputx):
        super(TestScatterDivNet, self).__init__()

        self.scattre_div = ops.ScatterDiv()
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        out = self.scattre_div(self.inputx, indices, updates)
        return out


def scattre_div_forward(nptype, expected):
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(nptype))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(nptype))

    net = TestScatterDivNet(inputx)
    output = net(indices, updates)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scattre_div_dynamic_updates():
    inputx = Tensor(np.ones((4, 2)).astype(np.float16))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    updates = Tensor(np.arange(4, 12).reshape((2, 2, 2)).astype(np.float16))
    updates_dy = Tensor(shape=(2, None, 2), dtype=mindspore.float16)

    net = TestScatterDivNet(inputx)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = np.array(
        [[0.25, 0.2], [0.1, 0.0909], [0.1666, 0.1428], [0.125, 0.1111]]
    ).astype(np.float16)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scattre_div_dynamic_indices():
    inputx = Tensor(np.ones((2, 3)).astype(np.float32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    indices_dy = Tensor(shape=(2, None), dtype=mindspore.int32)
    updates = Tensor(np.arange(1, 13).reshape((2, 2, 3)).astype(np.float32))

    net = TestScatterDivNet(inputx)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = np.array(
        [[1., 0.5, 0.33333334], [0.1, 0.09090909, 0.08333334]]
    ).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scattre_div_forward_float16():
    """
    Feature: test scattre_div forward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    expected = np.array([[0.000e+00, 9.418e-06, 1.764e-05],
                         [4.768e-07, 5.364e-07, 6.557e-07],
                         [0.000e+00, 0.000e+00, 0.000e+00]]).astype(np.float16)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_div_forward(np.float16, expected)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_div_forward(np.float16, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scattre_div_forward_float32():
    """
    Feature: test scattre_div forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    expected = np.array([[0.0000000e+00, 9.3984954e-06, 1.7640885e-05],
                         [4.5712085e-07, 5.6227748e-07, 6.4949910e-07],
                         [1.8554973e-08, 1.9582270e-08, 2.0286041e-08]]).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_div_forward(np.float32, expected)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_div_forward(np.float32, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scattre_div_dynamic_indices():
    """
    Feature: test scattre_div dynamic shape.
    Description: indices is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_div_dynamic_indices()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_div_dynamic_indices()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scattre_div_dynamic_updates():
    """
    Feature: test scattre_div dynamic shape.
    Description: updates is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scattre_div_dynamic_updates()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scattre_div_dynamic_updates()
