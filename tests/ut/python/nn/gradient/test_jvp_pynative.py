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
"""test jvp in pynative mode """

import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn.grad import Jvp

context.set_context(mode=context.PYNATIVE_MODE)

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


def test_jvp_single_input_single_output_default_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    Jvp(net)(x, v)


def test_jvp_single_input_single_output_custom_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    Jvp(net)(x, v)


def test_jvp_single_input_multiple_outputs_default_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    Jvp(net)(x, v)


def test_jvp_single_input_multiple_outputs_custom_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputNet()
    Jvp(net)(x, v)


def test_jvp_multiple_inputs_multiple_outputs_default_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    Jvp(net)(x, y, (v, v))


def test_jvp_multiple_inputs_multiple_outputs_custom_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputMultipleOutputNet()
    Jvp(net)(x, y, (v1, v2))


def test_jvp_multiple_inputs_single_output_default_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    Jvp(net)(x, y, (v, v))


def test_jvp_multiple_inputs_single_output_custom_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    v2 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    Jvp(net)(x, y, (v1, v2))


def test_jvp_wrong_input_v_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    with pytest.raises(TypeError):
        Jvp(net)(x, (v, v))


def test_jvp_wrong_input_v_2_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    with pytest.raises(TypeError):
        Jvp(net)(x, (v,))


def test_jvp_wrong_input_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    with pytest.raises(TypeError):
        Jvp(net)(x, x, v)


def test_jvp_wrong_input_2_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    with pytest.raises(TypeError):
        Jvp(net)((x, y), (v, v))


def test_jvp_wrong_input_3_pynative():
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputSingleOutputNet()
    with pytest.raises(TypeError):
        Jvp(net)(x, y, v)
