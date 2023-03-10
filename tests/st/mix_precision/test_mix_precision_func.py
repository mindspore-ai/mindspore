# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Test ms_class with mix_precision."""

import numpy as np
import pytest

import mindspore
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
import mindspore.amp as amp
import mindspore.context as context


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_static_loss_scaler(mode):
    """
    Feature: test StaticLossScaler.
    Description: `scale` and `unscale` are used to scale up loss value and scale down the grads.
    Expectation: the scaled and unscaled value are correct.
    """
    context.set_context(mode=mode)
    loss_scaler = amp.StaticLossScaler(scale_value=2**10)

    loss_value = Tensor([1.], mindspore.float32)
    scaled_loss_value = loss_scaler.scale(loss_value)
    grads = (Tensor(np.array([1.5, 1.0]), mindspore.float16),
             Tensor(np.array([1.2]), mindspore.float16))
    unscaled_grads = loss_scaler.unscale(grads)
    assert scaled_loss_value == loss_value * 1024
    assert (unscaled_grads[0] == grads[0] / 1024.).all()
    assert (unscaled_grads[1] == grads[1] / 1024.).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dynamic_loss_scaler(mode):
    """
    Feature: test DynamicLossScaler.
    Description: the `scale_value` can be adjusted dependent on whether grads are finite.
    Expectation: the `scale_value` can be adjusted correctly.
    """
    context.set_context(mode=mode)
    loss_scaler = amp.DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)

    grads = (Tensor(np.array([0.5, 1.0]), mindspore.float16),
             Tensor(np.array([0.2]), mindspore.float16))
    unscaled_grads = loss_scaler.unscale(grads)
    grads_finite = amp.all_finite(unscaled_grads)
    loss_scaler.counter = Parameter(Tensor(49, dtype=mstype.int32))
    loss_scaler.adjust(grads_finite)
    assert loss_scaler.scale_value.asnumpy() == np.array(2048.)

    grads = (Tensor(np.array([2., 1.0]), mindspore.float16),
             Tensor(np.array([0.2]), mindspore.float16))
    unscaled_grads = loss_scaler.unscale(grads)
    grads_finite = amp.all_finite(unscaled_grads)
    loss_scaler.scale_value = Parameter(Tensor(2**10, dtype=mstype.float32))
    loss_scaler.adjust(grads_finite)
    assert loss_scaler.scale_value.asnumpy() == np.array(1024.)
