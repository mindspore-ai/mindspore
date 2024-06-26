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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from tests.mark_utils import arg_mark


class MultiLabelSoftMarginLossNet(nn.Cell):
    def __init__(self, weight=None, reduction='mean'):
        super(MultiLabelSoftMarginLossNet, self).__init__()
        self.multilabel_soft_margin_loss = nn.MultiLabelSoftMarginLoss(weight=weight, reduction=reduction)

    def construct(self, x, target):
        return self.multilabel_soft_margin_loss(x, target)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('weight', [None, Tensor([1.0, 1.5, 0.8], mstype.float32)])
@pytest.mark.parametrize('reduction', ['mean', 'none', 'sum'])
def test_multilabel_soft_margin_loss(mode, weight, reduction):
    """
    Feature: MultiLabelSoftMarginLoss with weight=[None, Tensor([1.0, 1.5, 0.8], mstype.float32)],
    reduction=['mean', 'none', 'sum']
    Description: Verify the result of MultiLabelSoftMarginLoss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = MultiLabelSoftMarginLossNet(weight=weight, reduction=reduction)
    arr1 = np.array([[0.3, 0.6, 0.6], [0.9, 0.4, 0.2]], np.float32)
    arr2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], np.float32)
    x = Tensor(arr1, mstype.float32)
    label = Tensor(arr2, mstype.float32)
    output = net(x, label)
    if weight is None:
        if reduction == 'mean':
            expected = np.array(0.846940, np.float32)
        elif reduction == 'sum':
            expected = np.array(1.693880, np.float32)
        else:
            expected = np.array([0.776444, 0.917436], np.float32)
    else:
        if reduction == 'mean':
            expected = np.array(0.974961, np.float32)
        elif reduction == 'sum':
            expected = np.array(1.949922, np.float32)
        else:
            expected = np.array([0.920193, 1.029729], np.float32)
    assert np.allclose(output.asnumpy(), expected)
