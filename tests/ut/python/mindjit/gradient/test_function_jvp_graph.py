# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test function jvp in graph mode"""

import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.functional import jvp

context.set_context(mode=context.GRAPH_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3


class SingleInputMultipleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3, 2*x


class MultipleInputSingleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x + 3*y


class MultipleInputMultipleOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x, y**3


def test_jvp_single_input_single_output_default_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    jvp(net, x, v)


def test_jvp_single_input_single_output_custom_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with single input, single output and custom v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    jvp(net, x, v)


def test_jvp_single_input_multiple_outputs_default_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with single input, multiple outputs and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    jvp(net, x, v)


def test_jvp_single_input_multiple_outputs_custom_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with single input, multiple outputs and custom v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    jvp(net, x, v)


def test_jvp_multiple_inputs_single_output_default_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with multiple inputs, single output and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    jvp(net, (x, y), (v, v))


def test_jvp_multiple_inputs_single_output_custom_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with multiple inputs, single output and custom v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    jvp(net, (x, y), (v1, v2))


def test_jvp_multiple_inputs_multiple_outputs_default_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with multiple inputs, multiple outputs and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    jvp(net, (x, y), (v, v))


def test_jvp_multiple_inputs_multiple_outputs_custom_v_graph():
    """
    Features: Function jvp
    Description: Test jvp with multiple inputs, multiple outputs and custom v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    jvp(net, (x, y), (v1, v2))


def test_jvp_wrong_input_type_graph():
    """
    Features: Function jvp
    Description: Test jvp with wrong input type in graph mode.
    Expectation: No exception.
    """
    x = 1
    v = 1
    net = SingleInputSingleOutputNet()
    with pytest.raises(TypeError):
        jvp(net, x, v)
