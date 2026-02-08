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
"""test vjp in graph mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.functional import vjp

context.set_context(mode=context.GRAPH_MODE)


class SingleInputNet(nn.Cell):
    def construct(self, x):
        return x ** 3


class MultipleInputsOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2 * x, y ** 3


def test_vjp_single_input_graph():
    """
    Features: Function vjp
    Description: Test vjp with single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputNet()
    vjp(net, x)[1](v)


def test_vjp_multiple_inputs_default_v_graph():
    """
    Features: Function vjp
    Description: Test vjp with multiple input, single output and default v in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsOutputNet()
    vjp(net, x, y)[1](v, v)


def test_vjp_wrong_input_type_graph():
    """
    Features: Function vjp
    Description: Test vjp with wrong input type in graph mode.
    Expectation: No exception.
    """
    x = 1
    v = 1
    net = SingleInputNet()
    with pytest.raises(TypeError):
        vjp(net, x)[1](v)
