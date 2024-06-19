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
"""burgers pinns"""
import time
import pytest

import numpy as np
from mindspore import context, nn, ops, jit, set_seed, Tensor
from mindspore import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from src.burgers import Burgers1D
from src.utils import calculate_l2_error

set_seed(123456)
np.random.seed(123456)



class Net(nn.Cell):
    def __init__(self, in_channels=2, hidden_channels=128, out_channels=1):
        super().__init__()
        act = nn.Tanh()
        self.layers = nn.SequentialCell(
            nn.Dense(in_channels, hidden_channels, activation=act),
            nn.Dense(hidden_channels, hidden_channels, activation=act),
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
def test_mindflow_burgers_pinns():
    """
    Feature: burgers pinns
    Description: test train and eval
    Expectation: success
    """
    context.set_context(jit_level='O2')
    context.set_context(mode=context.GRAPH_MODE)
    model = Net()
    param_dict = load_checkpoint("/home/workspace/mindspore_ckpt/ckpt/burgers.ckpt")
    load_param_into_net(model, param_dict)
    optimizer = nn.Adam(model.trainable_params(), 0.0001)
    problem = Burgers1D(model)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    pde_data = Tensor([[-0.02541629, 0.12696983],
                       [0.30243418, 0.96671784],
                       [0.6249878, 0.260476],
                       [-0.61371905, 0.8972365],
                       [0.70554, 0.37674972]], mstype.float32)

    ic_data = Tensor([[0.1678119, 0.],
                      [-0.45064327, 0.],
                      [0.01379196, 0.],
                      [0.40799928, 0.],
                      [0.13942307, 0.]], mstype.float32)
    bc_data = Tensor([[1., 0.1909238],
                      [1., 0.70078486],
                      [-1., 0.70864534],
                      [1., 0.7291773],
                      [1., 0.30929238]], mstype.float32)
    inputs = Tensor([[-1., 0.],
                     [-0.99215686, 0.],
                     [-0.9843137, 0.],
                     [-0.9764706, 0.],
                     [-0.96862745, 0.]], mstype.float32)

    label = np.array([[1.22464680e-16],
                      [2.46374492e-02],
                      [4.92599411e-02],
                      [7.38525275e-02],
                      [9.84002783e-02]], np.float32)

    def forward_fn(pde_data, ic_data, bc_data):
        loss = problem.get_loss(pde_data, ic_data, bc_data)
        if use_ascend:
            loss = loss_scaler.scale(loss)

        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, ic_data, bc_data):
        loss, grads = grad_fn(pde_data, ic_data, bc_data)
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
        train_loss = train_step(pde_data, ic_data, bc_data)
        epoch_time = time.time() - time_beg
        print(f"epoch: {epoch} train loss: {train_loss} epoch time: {epoch_time}s")

    model.set_train(False)
    eval_error = calculate_l2_error(model, inputs, label, 5)
    print("eval_error:", eval_error)
    if context.get_context("device_target") == 'GPU':
        assert epoch_time < 0.03
    else:
        assert epoch_time < 0.02
    assert train_loss < 0.6
    assert eval_error < 0.8
