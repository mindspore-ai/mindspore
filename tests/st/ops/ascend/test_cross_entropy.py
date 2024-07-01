# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import torch
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Cell

from grad import GradOfFirstInput


class CrossEntropyLoss(Cell):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean',
                 label_smoothing_=0.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight, ignore_index, reduction, label_smoothing_)

    def construct(self, predict, target):
        return self.cross_entropy(predict, target)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def _test_cross_entropy_default_input_precision():
    """
    Feature: Test cross_entropy ops precision
    Description: Compare the forward result and grad result between mindspore and torch.
    Expectation: assert pass.
    """
    # prepare input
    context.set_context(device_target="Ascend")
    dtype = np.float32
    predict_shape = (3, 5)
    target_shape = (3,)
    predict_np = np.random.randn(*predict_shape).astype(dtype)
    target_np = np.random.randint(0, predict_shape[1], target_shape, dtype=np.int32)
    reduction = 'mean'
    ignore_index = -100
    label_smoothing = 0.0
    weight = None

    # calculate ms forward
    predict_tensor = Tensor(predict_np)
    target_tensor = Tensor(target_np)
    ms_net = CrossEntropyLoss(weight, ignore_index, reduction, label_smoothing)
    ms_infer_out = ms_net(predict_tensor, target_tensor)
    ms_forward = ms_infer_out.asnumpy()

    # calculate torch forward
    predict_tensor = torch.from_numpy(predict_np.copy().astype(np.float32))
    target_tensor = torch.from_numpy(target_np.copy()).long()
    torch_net = torch.nn.CrossEntropyLoss(weight, ignore_index=ignore_index, reduction=reduction)
    torch_infer_out = torch_net(predict_tensor, target_tensor)
    torch_forward = torch_infer_out.detach().numpy().astype(dtype)

    # compare ms forward and torch forward
    loss = 1e-4
    allclose_nparray(torch_forward, ms_forward, loss, loss)

    # calculate torch grad
    predict_tensor.requires_grad = True
    torch_net_2 = torch.nn.CrossEntropyLoss(weight, ignore_index=ignore_index, reduction=reduction)
    output = torch_net_2(predict_tensor, target_tensor)
    output_grad = torch.from_numpy(torch_forward.copy().astype(np.float32))
    output.backward(gradient=output_grad)
    torch_grad = predict_tensor.grad.detach().numpy()
    # calculate ms grad
    output_grad = Tensor(ms_forward)
    grad_net = GradOfFirstInput(ms_net)
    grad_net.set_train()
    predict_grad = Tensor(predict_np)
    target_grad = Tensor(target_np)
    input_grad = grad_net(predict_grad, target_grad, output_grad)
    ms_grad = input_grad.asnumpy()
    allclose_nparray(torch_grad.astype(dtype), ms_grad, loss, loss)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_default_input_pynative_mode():
    """
    Feature: Test cross_entropy ops infer and grad precision in pynative model.
    Description: Compare the forward result and grad result between mindspore and torch.
    Expectation: assert pass.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    _test_cross_entropy_default_input_precision()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_default_input_graph_mode():
    """
    Feature: Test cross_entropy ops infer and grad precision in graph model.
    Description: Compare the forward result and grad result between mindspore and torch.
    Expectation: assert pass.
    """
    context.set_context(mode=context.GRAPH_MODE)
    _test_cross_entropy_default_input_precision()
