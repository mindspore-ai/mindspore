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
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G


class ConcatOffsetNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.unique = P.Unique()
        self.concat_offset = G.ConcatOffset(3, 0)
        self.reshape = P.Reshape()

    def construct(self, x, y, z):
        x = self.reshape(self.unique(x)[0], (-1, 1, 2, 1))
        y = self.reshape(self.unique(y)[0], (-1, 1, 2, 1))
        z = self.reshape(self.unique(z)[0], (-1, 1, 2, 1))
        out = self.concat_offset((x, y, z))
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_offset_dynamic_gpu():
    """
    /// Feature: Concatoffset op dynamic shape
    /// Description: Concatoffset forward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    x = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    x2 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    x3 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    net = ConcatOffsetNet()
    out = net(x, x2, x3)
    expect = np.array([[0, 0, 0, 0],
                       [3, 0, 0, 0],
                       [6, 0, 0, 0]])
    if isinstance(out, tuple):
        assert (np.array(out) == expect).all()
    else:
        assert (out.asnumpy() == expect).all()
