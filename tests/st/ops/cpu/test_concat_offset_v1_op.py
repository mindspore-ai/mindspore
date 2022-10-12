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

import sys
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops.operations.array_ops import ConcatOffsetV1


class ConcatOffsetNetV1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.concat_offset_v1 = ConcatOffsetV1()

    def construct(self, axis, x, y, z):
        out = self.concat_offset_v1(axis, (x, y, z))
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_offset_v1():
    """
    /// Feature: ConcatOffsetNetV1 op dynamic shape
    /// Description: ConcatOffsetNetV1 forward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="CPU", save_graphs=True)

    axis = Tensor(1, dtype=mstype.int32)
    x1 = Tensor(np.array([1, 2, 3]).astype(np.int32))
    x2 = Tensor(np.array([1, 5, 3]).astype(np.int32))
    x3 = Tensor(np.array([1, 4, 3]).astype(np.int32))

    net = ConcatOffsetNetV1()
    out = net(axis, x1, x2, x3)
    expect = np.array([[0, 0, 0],
                       [0, 2, 0],
                       [0, 7, 0]])
    if isinstance(out, (list, tuple)):
        assert (np.array(out) == expect).all()
    else:
        assert (out.asnumpy() == expect).all()
