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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations.array_ops import ArgminV2
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class ArgMinV2DynatimicShape(nn.Cell):
    def __init__(self, gather_axis=1, argmin_axis=1):
        super(ArgMinV2DynatimicShape, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.argmin = ArgminV2()
        self.gather_axis = gather_axis
        self.argmin_axis = argmin_axis

    def construct(self, x, indices):
        unique_index, _ = self.unique(indices)
        x = self.gather(x, unique_index, self.gather_axis)
        y = self.argmin(x, self.argmin_axis)
        return y


def test_argmin_v2_base():
    """
    Feature: Test ArgminV2 op in with dynamic shape input.
    Description: Input 2D Tensor.
    Expectation: Successful execution.
    """
    x = Tensor(np.array([[4, 8, 1, 6], [4, 3, 6, 2], [4, 4, 1, 1],
                         [2, 4, 4, 8], [8, 7, 8, 9], [9, 7, 2, 9]]).astype(np.float32))
    index = Tensor([1, 2, 1], dtype=mindspore.int32)
    net = ArgMinV2DynatimicShape()
    res = net(x, index)
    expect = np.array([1, 0, 1, 0, 0, 1]).astype(np.int32)
    assert np.allclose(res.asnumpy(), expect)
