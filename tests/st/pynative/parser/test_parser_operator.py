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
""" test_parser_operator """
import pytest
import numpy as np
from mindspore import context
from mindspore.nn import ReLU
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parser_operator_floor_div():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ReLU()

        def construct(self, x):
            x = self.relu(x)
            x = 3 // x
            return x

    input_np_x = np.array(2).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me_x)

    assert np.allclose(out_me.asnumpy(), 3 // input_np_x, 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_pow():
    """
    Feature: tensor pow
    Description: Test tensor pow in pynative
    Expectation: No exception.
    """
    x = (2, 2)
    y_np = np.array([8.0, 8.0]).astype(np.float32)
    y = Tensor(y_np)
    out_me = x ** y
    out_np = np.array(x) ** y_np
    assert np.allclose(out_me.asnumpy(), out_np, rtol=0.01, atol=0.01)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_matmul():
    """
    Feature: tensor matmul
    Description: Test tensor matmul in pynative
    Expectation: No exception.
    """
    x_np = np.arange(2*3*4).reshape(2, 3, 4).astype('float32')
    y_np = np.arange(4*5).reshape(4, 5).astype('float32')
    x = Tensor(x_np)
    y = Tensor(y_np)
    out_me = x @ y
    out_np = x_np @ y_np
    assert np.allclose(out_me.asnumpy(), out_np, rtol=0.01, atol=0.01)
