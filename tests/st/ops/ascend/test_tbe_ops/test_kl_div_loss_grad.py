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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap


class Net(nn.Cell):
    def __init__(self, reduction=None):
        super(Net, self).__init__()
        if reduction is not None:
            self.kl_div_loss_grad = G.KLDivLossGrad(reduction)
        else:
            self.kl_div_loss_grad = G.KLDivLossGrad()

    def construct(self, x, y, dy):
        return self.kl_div_loss_grad(dy, x, y)

reduction_list = ["none", "mean", "batchmean", "sum"]
expect_list = [[[0, 0], [-1, 0]],
               [[0, 0.25], [0.25, 0]],
               [[0, 0.5], [0.5, 0]],
               [[0, 1], [1, 0]]]


def generate_test_cases(dtype, mode, reduction):
    context.set_context(mode=mode, device_target="Ascend")
    prediction = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    if reduction == "none":
        dy = Tensor(np.array([[-1, 0], [1, 1]]).astype(dtype))
    else:
        dy = Tensor(np.array([-1]).astype(dtype))
    backward_net = Net(reduction)
    output = backward_net(prediction, target, dy)
    expect = np.array(expect_list[reduction_list.index(reduction)])
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ["none", "mean", "batchmean", "sum"])
def test_kl_div_loss_grad_with_static_input(data_type, run_mode, reduction):
    """
    Feature: KLDivLossGrad operators.
    Description: KLDivLossGrad with different mode.
    Expectation: run success without error
    """
    generate_test_cases(data_type, run_mode, reduction)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_vmap_case():
    """
    Feature: KLDivLossGrad with vmap mode.
    Description: KLDivLossGrad with vmap mode, 2d input.
    Expectation: run success without error.
    """
    class NetVmap(nn.Cell):
        def __init__(self, reduction="none"):
            super(NetVmap, self).__init__()
            if reduction is not None:
                self.kl_div_loss_grad = G.KLDivLossGrad(reduction)
            else:
                self.kl_div_loss_grad = G.KLDivLossGrad()

        def construct(self, dy, x, y):
            return self.kl_div_loss_grad(dy, x, y)

    class WrapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, x, y, dy):
            return vmap(self.net, self.in_axes, self.out_axes)(dy, x, y)

    dtype = np.float32
    prediction = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]).astype(dtype))
    target = Tensor(np.array([[-1, 1], [1, -1]]).astype(dtype))
    dy = Tensor(np.array([[-1, 0], [1, 1]]).astype(dtype))
    output = WrapNet(NetVmap(), 0, 0)(prediction, target, dy)
    print(output)
    expect = np.array([[0, 0], [-1, 0]])
    assert np.allclose(output.asnumpy(), expect)
