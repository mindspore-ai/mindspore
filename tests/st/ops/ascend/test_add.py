# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


x = np.ones([1, 3, 3, 4]).astype(np.float32)
y = np.ones([1, 3, 3, 4]).astype(np.float32)


def test_net():
    add = Net()
    output = add(Tensor(x), Tensor(y))
    print(x)
    print(y)
    print(output.asnumpy())


def test_add_tensor_api(nptype=np.float32, mstype=None):
    """
    Feature: test add tensor api.
    Description: test inputs given their dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.array([1, 2, 3]).astype(nptype))
    input_y = Tensor(np.array([4, 5, 6]).astype(nptype))
    if mstype == mindspore.bfloat16:
        input_x = Tensor(np.array([1, 2, 3]).astype(nptype), mindspore.bfloat16)
        input_y = Tensor(np.array([4, 5, 6]).astype(nptype), mindspore.bfloat16)
    output = input_x.add(input_y)
    expected = np.array([5, 7, 9]).astype(np.int32)
    if mstype == mindspore.bfloat16:
        expected = np.array([5, 7, 9]).astype(np.float32)
        np.testing.assert_array_almost_equal(output.float().asnumpy(), expected)
    else:
        np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_add_float32_tensor_api():
    """
    Feature: test add tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_add_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_add_tensor_api(np.float32)



@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_add_bfloat16_tensor_api():
    """
    Feature: test add tensor api.
    Description: test bfloat16 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_add_tensor_api(np.float32, mstype=mindspore.bfloat16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_add_tensor_api(np.float32, mstype=mindspore.bfloat16)
