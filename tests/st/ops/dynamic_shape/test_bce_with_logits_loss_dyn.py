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

import math
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def __init__(self, reduction):
        super(Net, self).__init__()
        self.loss = P.BCEWithLogitsLoss(reduction=reduction)

    def construct(self, predict, target, weight, pos_weight):
        return self.loss(predict, target, weight, pos_weight)


def net_run():
    predict = Tensor(np.arange(6).reshape(2, 3).astype(np.float16))
    target = Tensor(np.arange(34, 40).reshape(2, 3).astype(np.float16))
    weight = Tensor(np.array([2, 3, 1]).astype(np.float16))
    pos_weight = Tensor(np.array([6, 3, 4]).astype(np.float16))
    net = Net("mean")
    net.set_inputs(Tensor(shape=[None, None], dtype=mstype.float16),
                   Tensor(target), Tensor(weight), Tensor(pos_weight))
    output = net(predict, target, weight, pos_weight)
    expected = -113.55404
    # assert scalar
    assert math.isclose(output.asnumpy().tolist(), expected, rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bce_mean_dyn_ascend():
    """
    Feature: Test dynamic shape of BCEWithLogitsLoss op that the reduction is mean on ascend.
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net_run()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bce_mean_dyn_ascend_pynative():
    """
    Feature: Test dynamic shape of BCEWithLogitsLoss op that the reduction is mean on ascend.
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    net_run()
