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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad


class TileNet(nn.Cell):
    def __init__(self):
        super(TileNet, self).__init__()
        self.tile = P.Tile()

    def construct(self, x, multiples):
        return self.tile(x, multiples)


def dynamic_shape():
    test_dynamic = TestDynamicGrad(TileNet())
    x = Tensor(np.array([[0], [1], [2], [3]]).astype(np.float32))
    y = (1, 4, 2)
    test_dynamic.test_dynamic_grad_net((x, y))


def dynamic_rank():
    test_dynamic = TestDynamicGrad(TileNet())
    x = Tensor(np.array([[0], [1], [2], [3]]).astype(np.float32))
    y = (1, 4, 2)
    test_dynamic.test_dynamic_grad_net((x, y), True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_tile():
    """
    Feature: Tile Grad DynamicShape.
    Description: Test case of dynamic shape for Tile grad operator.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    dynamic_shape()
    dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    dynamic_shape()
    dynamic_rank()
