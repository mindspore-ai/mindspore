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
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def construct(self, x, y):
        return ops.soft_margin_loss(x, y, self.reduction)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_soft_margin_loss(mode, reduction):
    """
    Feature: ops.soft_margin_loss
    Description: Verify the result of ops.soft_margin_loss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(reduction)
    a = Tensor(np.arange(8).reshape((4, 2)), mstype.float32)
    b = Tensor(np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2)), mstype.float32)
    output = net(a, b)
    if reduction == 'none':
        expected = np.array([[6.9315e-01, 4.8587e-02], [1.2693e-01, 6.1442e-06],
                             [3.7751e-11, 2.0612e-09], [3.7751e-11, 4.3596e-28]]).astype(np.float32)
    elif reduction == 'mean':
        expected = np.array(0.1086)
    else:
        expected = np.array(0.8687)
    assert np.allclose(output.asnumpy(), expected, atol=0.0001, rtol=0.0001)
