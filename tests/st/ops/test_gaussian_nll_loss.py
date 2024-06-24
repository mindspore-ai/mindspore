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
from mindspore import Tensor, ops, nn
from mindspore import context


class Net(nn.Cell):
    def __init__(self, full=False, reduction='mean'):
        super(Net, self).__init__()
        self.full = full
        self.reduction = reduction

    def construct(self, x, label, v):
        loss = ops.gaussian_nll_loss(x, label, v, full=self.full, reduction=self.reduction)
        return loss


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('full', [True, False])
def test_gaussian_nll_loss_full(mode, full):
    """
    Feature: GaussianNLLLoss with reduction='mean'
    Description: Verify the result of GaussianNLLLoss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(full=full)
    arr1 = np.arange(8).reshape((4, 2))
    arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    var = Tensor(np.ones((4, 1)), mstype.float32)
    output = net(a, b, var)
    if full:
        expected = np.array(2.35644, np.float32)
    else:
        expected = np.array(1.4375, np.float32)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_gaussian_nll_loss_reduction(mode, reduction):
    """
    Feature: GaussianNLLLoss with full=False
    Description: Verify the result of GaussianNLLLoss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(reduction=reduction)
    arr1 = np.arange(8).reshape((4, 2))
    arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    var = Tensor(np.ones((4, 1)), mstype.float32)
    output = net(a, b, var)
    if reduction == 'mean':
        expected = np.array(1.4375, np.float32)
    elif reduction == 'sum':
        expected = np.array(11.5, np.float32)
    else:
        expected = np.array([[2.0000, 2.0000], [0.5000, 0.5000],
                             [2.0000, 0.5000], [2.0000, 2.0000]], np.float32)
    assert np.allclose(output.asnumpy(), expected)
