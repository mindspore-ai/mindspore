# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, nn
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax = P.Softmax(axis=-1)

    def construct(self, x):
        return self.softmax(x)


def test_net():
    x = np.random.randn(32, 10).astype(np.float32)
    softmax = Net()
    output = softmax(Tensor(x))
    print(x)
    print(output.asnumpy())


def run_softmax_api(ms_type, nptype):
    """
    Feature: test softmax tensor api.
    Description: test inputs using given dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), ms_type)
    softmax = Net()
    output = softmax(input_x)
    print("ms output:", output)
    excepted = np.array([0.01165623, 0.03168492, 0.08612854, 0.23412165, 0.6364086]).astype(nptype)
    if ms_type == mindspore.bfloat16:
        np.testing.assert_array_almost_equal(output.float().asnumpy(), excepted, decimal=3)
    else:
        np.testing.assert_array_almost_equal(output.asnumpy(), excepted, decimal=6)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_softmax_float32_tensor_api():
    """
    Feature: test softmax tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_softmax_api(mindspore.float32, np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_softmax_api(mindspore.float32, np.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_softmax_bfloat16_tensor_api():
    """
    Feature: test softmax tensor api.
    Description: test bfloat16 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_softmax_api(mindspore.bfloat16, np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    run_softmax_api(mindspore.bfloat16, np.float32)
