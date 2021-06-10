# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, reduction):
        super(Net, self).__init__()
        self.loss = P.NLLLoss(reduction=reduction)

    def construct(self, predict, target, weight):
        return self.loss(predict, target, weight)


def nll_loss_template(nptype_input, nptype_weight, reduction):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    nll_loss_net = Net(reduction)

    predict = Tensor(np.array([[0.53, 0.74, -2.12], [1.29, -0.34, -1.13]]).astype(nptype_input))

    target = Tensor(np.array([0, 1]).astype(np.int32))

    weight = Tensor(np.array([0.45, -0.32, 1.21]).astype(nptype_weight))

    loss, total_weight = nll_loss_net(predict, target, weight)

    loss_np = loss.asnumpy()
    total_weight_np = total_weight.asnumpy()

    expected_tot_weight = np.array(0.129999995)

    if reduction == 'none':
        expected_loss = np.array([-0.238499984, -0.108800001])
    elif reduction == 'mean':
        expected_loss = np.array(-2.67153859)
    elif reduction == 'sum':
        expected_loss = np.array(-0.347299993)

    if nptype_input == np.float32 and nptype_weight == np.float32:
        ertol_loss = 1e-06
    elif nptype_input == np.float16 or nptype_weight == np.float16:
        ertol_loss = 1e-03

    if nptype_weight == np.float32:
        ertol_weight = 1e-06
    elif nptype_weight == np.float16:
        ertol_weight = 1e-03

    np.testing.assert_allclose(loss_np, expected_loss, ertol_loss)
    np.testing.assert_allclose(total_weight_np, expected_tot_weight, ertol_weight)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nll_loss_no_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_template(np.float32, np.float32, "none")
    nll_loss_template(np.float32, np.float16, "none")
    nll_loss_template(np.float16, np.float32, "none")
    nll_loss_template(np.float16, np.float16, "none")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nll_loss_mean_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_template(np.float32, np.float32, "mean")
    nll_loss_template(np.float32, np.float16, "mean")
    nll_loss_template(np.float16, np.float32, "mean")
    nll_loss_template(np.float16, np.float16, "mean")

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nll_loss_sum_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_template(np.float32, np.float32, "sum")
    nll_loss_template(np.float32, np.float16, "sum")
    nll_loss_template(np.float16, np.float32, "sum")
    nll_loss_template(np.float16, np.float16, "sum")
