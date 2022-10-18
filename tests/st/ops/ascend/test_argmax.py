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
from mindspore.common.api import jit
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.argmax = P.Argmax(axis=1)

    @jit
    def construct(self, x):
        return self.argmax(x)


def test_net():
    x = np.random.randn(32, 10).astype(np.float32)
    argmax = Net()
    output = argmax(Tensor(x))
    print(x)
    print(output.asnumpy())


def adaptive_argmax_functional(nptype):
    x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(nptype))
    output = F.argmax(x, axis=-1, output_type=mstype.int32)
    expected = np.array([1, 0, 0]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_argmax_float32_functional():
    """
    Feature: test argmax functional api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    adaptive_argmax_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    adaptive_argmax_functional(np.float32)
