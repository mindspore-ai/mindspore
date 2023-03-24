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

import random
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import IdentityN


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.identity_n = IdentityN()

    def construct(self, x):
        return self.identity_n(x)


def generate_testcases(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(3, 4, 5, 6).astype(nptype)
    net = Net()
    input_tensor = [Tensor(x) for i in range(random.randint(2, 10))]
    output = net(input_tensor)
    np.testing.assert_almost_equal([el.asnumpy() for el in output], [el.asnumpy() for el in input_tensor])
    assert id(input_tensor) != id(output)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.random.randn(3, 4, 5, 6).astype(nptype)
    net = Net()
    input_tensor = [Tensor(x) for i in range(random.randint(1, 10))]
    output = net(input_tensor)
    np.testing.assert_almost_equal([el.asnumpy() for el in output], [el.asnumpy() for el in input_tensor])
    assert id(input_tensor) != id(output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_identity_n_float32_input_error():
    """
    Feature: test IdentityN forward.
    Description: test float32 inputs.
    Expectation: raise TypeError.
    """
    nptype = np.float32
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.randn(3, 4).astype(nptype)
    net = Net()
    input_tensor = Tensor(x)
    with pytest.raises(TypeError):
        output = net(input_tensor)
        np.testing.assert_almost_equal([el.asnumpy() for el in output], [el.asnumpy() for el in input_tensor])
        assert id(input_tensor) != id(output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_identity_n_float32():
    """
    Feature: test IdentityN forward.
    Description: test float32 inputs.
    Expectation: success
    """
    generate_testcases(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_identity_n_int32():
    """
    Feature: test IdentityN forward.
    Description: test int32 inputs.
    Expectation: success
    """
    generate_testcases(np.int32)
