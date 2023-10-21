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
from mindspore.ops.operations import nn_ops as P
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self, reduction):
        super(Net, self).__init__()
        self.loss = P.MultilabelMarginLoss(reduction=reduction)

    def construct(self, x, target):
        return self.loss(x, target)


class GradNet(nn.Cell):
    def __init__(self, reduction):
        super(GradNet, self).__init__()
        self.grad = G.MultilabelMarginLossGrad(reduction=reduction)

    def construct(self, y_grad, x, target, is_target):
        gout = self.grad(y_grad, x, target, is_target)
        return gout


def multilabel_margin_loss_template(nptype_input, reduction):
    multilabel_margin_loss_net = Net(reduction)

    pt_dic = {
        0: [
            [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.4, 0.8]],
            [[2, 1, -1, 1], [3, 0, -1, 1]],
        ],
        1: [
            [0.1, 0.2, 0.3, 0.4],
            [2, 1, -1, 1],
        ]
    }

    answer_dic = {
        0: {
            'none': [1, 0.85], 'sum': [1.85], 'mean': [0.925],
            'tar': [[0, 1, 1, 0], [1, 0, 0, 1]],
        },
        1: {
            'none': [1], 'sum': [1], 'mean': [1],
            'tar': [0, 1, 1, 0],
        }
    }

    for idx in range(2):
        predict = Tensor(np.array(pt_dic.get(idx)[0]).astype(nptype_input))
        target = Tensor(np.array(pt_dic.get(idx)[1]).astype(np.int32))
        loss, is_target = multilabel_margin_loss_net(predict, target)
        loss_np = loss.asnumpy()
        expected_loss = np.array(
            answer_dic.get(idx)[reduction]).astype(nptype_input)
        expected_tar = Tensor(np.array(answer_dic.get(idx)['tar']).astype(np.int32))

        if nptype_input == np.float64:
            ertol_loss = 1e-05
        elif nptype_input == np.float32:
            ertol_loss = 1e-04
        elif nptype_input == np.float16:
            ertol_loss = 1e-03

        np.testing.assert_allclose(loss_np, expected_loss, ertol_loss)
        assert loss_np.dtype == expected_loss.dtype
        assert is_target.dtype == expected_tar.dtype


def multilabel_margin_loss_grad_template(nptype_input, reduction):
    multilabel_margin_loss_grad_net = GradNet(reduction)

    p_t_it_dic = {
        0: [
            [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.4, 0.8]],
            [[2, 1, -1, 1], [3, 0, -1, 1]],
            [[0, 1, 1, 0], [1, 0, 0, 1]],
        ],
        1: [
            [0.1, 0.2, 0.3, 0.4],
            [2, 1, -1, 1],
            [0, 1, 1, 0],
        ]
    }

    yg_xg_dic = {
        0: {
            'none': [[0.1, 0.2], [[0.0500, -0.0500, -0.0500, 0.0500], [-0.1000, 0.1000, 0.1000, -0.1000]]],
            'sum': [[0.3], [[0.1500, -0.1500, -0.1500, 0.1500], [-0.1500, 0.1500, 0.1500, -0.1500]]],
            'mean': [[0.4], [[0.1000, -0.1000, -0.1000, 0.1000], [-0.1000, 0.1000, 0.1000, -0.1000]]],
        },
        1: {
            'none': [[0.1], [0.0500, -0.0500, -0.0500, 0.0500]],
            'sum': [[0.2], [0.1000, -0.1000, -0.1000, 0.1000]],
            'mean': [[0.3], [0.1500, -0.1500, -0.1500, 0.1500]],
        }
    }

    for idx in range(2):
        predict = Tensor(np.array(p_t_it_dic.get(idx)[0]).astype(nptype_input))
        target = Tensor(np.array(p_t_it_dic.get(idx)[1]).astype(np.int32))
        is_target = Tensor(np.array(p_t_it_dic.get(idx)[2]).astype(np.int32))
        y_grad = Tensor(np.array(yg_xg_dic.get(
            idx)[reduction][0]).astype(nptype_input))
        x_grad = multilabel_margin_loss_grad_net(
            y_grad, predict, target, is_target)
        x_grad_np = x_grad.asnumpy()
        expected_x_grad = np.array(yg_xg_dic.get(
            idx)[reduction][1]).astype(nptype_input)
        if nptype_input == np.float64:
            ertol_loss = 1e-05
        elif nptype_input == np.float32:
            ertol_loss = 1e-04
        elif nptype_input == np.float16:
            ertol_loss = 1e-03
        np.testing.assert_allclose(x_grad_np, expected_x_grad, ertol_loss)
        assert x_grad_np.dtype == expected_x_grad.dtype


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multilabel_margin_loss_graph():
    """
    Feature: reduction = none
    Description: all dtype of input with all reduction
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    for reduction in ['none', 'sum', 'mean']:
        multilabel_margin_loss_template(np.float32, reduction)
        multilabel_margin_loss_template(np.float16, reduction)
        multilabel_margin_loss_template(np.float64, reduction)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multilabel_margin_loss_grad_graph():
    """
    Feature: reduction = none
    Description: the grad of loss, for all dtype of input with all reduction
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    for reduction in ['none', 'sum', 'mean']:
        multilabel_margin_loss_grad_template(np.float32, reduction)
        multilabel_margin_loss_grad_template(np.float16, reduction)
        multilabel_margin_loss_grad_template(np.float64, reduction)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multilabel_margin_loss_pynative():
    """
    Feature: reduction = none
    Description: all dtype of input with all reduction
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    for reduction in ['none', 'sum', 'mean']:
        multilabel_margin_loss_template(np.float32, reduction)
        multilabel_margin_loss_template(np.float16, reduction)
        multilabel_margin_loss_template(np.float64, reduction)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multilabel_margin_loss_grad_pynative():
    """
    Feature: reduction = none
    Description: the grad of loss, for all dtype of input with all reduction
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    for reduction in ['none', 'sum', 'mean']:
        multilabel_margin_loss_grad_template(np.float32, reduction)
        multilabel_margin_loss_grad_template(np.float16, reduction)
        multilabel_margin_loss_grad_template(np.float64, reduction)
