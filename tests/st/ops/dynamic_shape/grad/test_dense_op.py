# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops.operations import nn_ops as ops

from .test_grad_of_dynamic import TestDynamicGrad


class Dense(nn.Cell):
    def __init__(self):
        super(Dense, self).__init__()
        self.dense = ops.Dense()

    def construct(self, x, w, b):
        x = self.dense(x, w, b)
        return x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_dense_op():
    """
    Feature: Dynamic Dense gpu kernel
    Description: test the rightness of Dense gpu kernel in dynamic shape and rank
    Expectation: the result match with static shape
    """
    m, n, k = 5, 3, 4
    x = Tensor(np.random.random((m, k)).astype(np.float32))
    w = Parameter(np.random.random((n, k)).astype(np.float32))
    b = Parameter(np.random.random((n,)).astype(np.float32))
    dense = Dense()
    test_dynamic_grad = TestDynamicGrad(dense)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_dynamic_grad.test_dynamic_grad_net((x, w, b), False)
    test_dynamic_grad.test_dynamic_grad_net((x, w, b), True)
