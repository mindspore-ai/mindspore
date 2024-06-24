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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from mindspore import nn, context, Tensor
from mindspore.ops import operations as P
from .test_grad_of_dynamic import TestDynamicGrad


class NetAdd(nn.Cell):
    def __init__(self):
        super(NetAdd, self).__init__()
        self.add = P.Add()

    def construct(self, x, y):
        return self.add(x, y)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetAdd())
    test_dynamic_bc = TestDynamicGrad(NetAdd())
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.float32))
    y = Tensor(np.array([[-1, 2, -3, 0], [8, 6, -9, 1], [8, 10, 0, 12]]).astype(np.float32))
    z = Tensor(np.array([7]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net((x, y), is_dynamic_rank)
    test_dynamic_bc.test_dynamic_grad_net((x, z), is_dynamic_rank)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_dynamic_shape():
    """
    Feature: test Add dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_grad_dynamic_rank():
    """
    Feature: test Add dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)
