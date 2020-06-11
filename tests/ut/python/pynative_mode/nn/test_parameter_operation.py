# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_tensor_operation """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)

def test_parameter_add():
    x = Parameter(Tensor(np.ones((3, 3)).astype(np.float32)), name="ref")
    y = Tensor(np.ones((3, 3)).astype(np.float32))
    expect = np.ones((3, 3)).astype(np.float32) * 2
    z = x + y
    assert np.allclose(z.asnumpy(), expect)

def test_parameter_sub():
    x = Parameter(Tensor(np.ones((3, 3)).astype(np.float32) * 2), name="ref")
    y = Tensor(np.ones((3, 3)).astype(np.float32))
    expect = np.ones((3, 3)).astype(np.float32)
    z = x - y
    assert np.allclose(z.asnumpy(), expect)

def test_parameter_mul():
    x = Parameter(Tensor(np.ones((3, 3)).astype(np.float32) * 2), name="ref")
    y = Tensor(np.ones((3, 3)).astype(np.float32) * 2)
    expect = np.ones((3, 3)).astype(np.float32) * 4
    z = x * y
    assert np.allclose(z.asnumpy(), expect)

def test_parameter_div():
    x = Parameter(Tensor(np.ones((3, 3)).astype(np.float32) * 8), name="ref")
    y = Tensor(np.ones((3, 3)).astype(np.float32) * 2)
    expect = np.ones((3, 3)).astype(np.float32) * 4
    z = x / y
    assert np.allclose(z.asnumpy(), expect)

class ParameterNet(nn.Cell):
    def __init__(self):
        super(ParameterNet, self).__init__()
        self.weight = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)), name="ref")

    def construct(self, x):
        self.weight = x

def test_parameter_assign():
    """test parameter assign with tensor"""
    input_x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 8.0]], np.float32))
    net = ParameterNet()
    net(input_x)
    assert np.allclose(net.weight.data.asnumpy(), input_x.asnumpy())
