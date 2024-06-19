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
"""navier stokes pinns"""
import time
import pytest

import numpy as np
from mindspore import context, nn, ops, jit, set_seed, Tensor
import mindspore.common.dtype as mstype

from src.navier_stokes2d import NavierStokes2D

from tests.st.utils import test_utils

set_seed(123456)
np.random.seed(123456)


class Net(nn.Cell):
    """MLP"""

    def __init__(self, in_channels=2, hidden_channels=128, out_channels=1):
        super().__init__()
        act = nn.Tanh()
        self.layers = nn.SequentialCell(
            nn.Dense(in_channels, hidden_channels, activation=act),
            nn.Dense(hidden_channels, hidden_channels, activation=act),
            nn.Dense(hidden_channels, hidden_channels, activation=act),
            nn.Dense(hidden_channels, out_channels)
        )

    def construct(self, x):
        return self.layers(x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_mindflow_navier_stokes():
    """
    Feature: navier_stokes pinns
    Description: test train
    Expectation: success
    """
    context.set_context(jit_level='O2')
    context.set_context(mode=context.GRAPH_MODE)
    model = Net(in_channels=3, out_channels=3)
    optimizer = nn.Adam(model.trainable_params(), 0.0001)
    problem = NavierStokes2D(model)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    pde_data = Tensor([[4.4814544, -1.6147294, 3.8416946],
                       [5.8124804, -0.49786586, 5.0063257],
                       [1.1191559, 0.9042227, 4.2193437],
                       [1.9051491, 1.1916666, 3.8141823],
                       [2.8169591, 0.3456305, 2.9655836]], mstype.float32)
    bc_data = Tensor([[3.8282828, -2., 1.4],
                      [6.3030305, -2., 1.4],
                      [1., 1.1020408, 3.6],
                      [5.8080807, -2., 6.7],
                      [3.5454545, -2., 4.]], mstype.float32)
    ic_data = Tensor([[6.6565657, -0.2857143, 0.],
                      [7.7171717, 1.1836735, 0.],
                      [2.6262627, -0.4489796, 0.],
                      [2.909091, 1.3469387, 0.],
                      [1.2121212, -0.04081633, 0.]], mstype.float32)
    bc_label = Tensor([[0.9934991, -0.06462386],
                       [1.0738759, 0.14259282],
                       [1.2844703, -0.06666262],
                       [1.0665872, 0.14132853],
                       [1.0729637, 0.1342909]], mstype.float32)
    ic_label = Tensor([[0.6541988, 0.26443157, -0.0937218],
                       [1.0388365, -0.32561874, -0.04165602],
                       [0.13536283, 0.00210919, -0.06288609],
                       [1.0713681, -0.24921523, -0.08316418],
                       [-0.20542848, 0.19257492, -0.39120102]], mstype.float32)

    def forward_fn(pde_data, bc_data, bc_label, ic_data, ic_label):
        loss = problem.get_loss(pde_data, bc_data, bc_label, ic_data, ic_label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data, bc_label, ic_data, ic_label):
        loss, grads = grad_fn(pde_data, bc_data, bc_label, ic_data, ic_label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)

        loss = ops.depend(loss, optimizer(grads))
        return loss
    epochs = 10
    for epoch in range(1, 1 + epochs):
        model.set_train(True)
        time_beg = time.time()
        train_loss = train_step(pde_data, bc_data, bc_label, ic_data, ic_label)
        epoch_time = time.time() - time_beg
        print(
            f"epoch: {epoch} train loss: {train_loss} epoch time: {epoch_time}s")
    model.set_train(False)

    if context.get_context("device_target") == 'GPU':
        assert epoch_time < 0.015
    else:
        assert epoch_time < 0.01
    assert train_loss < 0.8
