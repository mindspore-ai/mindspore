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
from tests.mark_utils import arg_mark
import torch
import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.common.dtype as mstype


class NetTripletMarginLoss(nn.Cell):
    def __init__(self, margin=Tensor(1.0, mstype.float32), p=2, swap=False, eps=1e-6, reduction="mean"):
        super(NetTripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.swap = swap
        self.eps = eps
        self.reduction = reduction

    def construct(self, anchor, positive, negative):
        return ops.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                       eps=self.eps, swap=self.swap, reduction=self.reduction)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_triplet_margin_loss_float64(mode):
    """
    Feature: Input type of float64
    Description: Input type of [float64, float64, float64].
    Expectation: success.
    """
    context.set_context(mode=mode)
    data_type = np.float64
    anchor_array = np.array([[1.3, 20.5, 5.6],
                             [3.5, 4.8, 7.2],
                             [0.2, 0.01, 1],
                             [4, 4.1, 20]]).astype(data_type)
    positive_array = np.array([[2., 10., 1.],
                               [6., 7., 10.],
                               [13., 4., 1.],
                               [0.33, -4, -1.5]]).astype(data_type)
    negative_array = np.array([[2., 21., 6.],
                               [68., 9., 10.],
                               [131., 25., 16.],
                               [0.31, -0.14, -16.]]).astype(data_type)
    margin = np.float32(2.0)
    p = 0
    swap = True
    reduction = "none"
    eps = 1e-5

    anchor = Tensor(anchor_array)
    positive = Tensor(positive_array)
    negative = Tensor(negative_array)
    triplet_margin_loss = NetTripletMarginLoss(margin=margin, p=p, eps=eps,
                                               swap=swap, reduction=reduction)
    output_ms = triplet_margin_loss(anchor, positive, negative)

    torch_anchor = torch.tensor(anchor_array)
    torch_positive = torch.tensor(positive_array)
    torch_negative = torch.tensor(negative_array)
    expect = torch.nn.functional.triplet_margin_loss(torch_anchor, torch_positive,
                                                     torch_negative, margin=margin,
                                                     p=p, eps=eps, swap=swap,
                                                     reduction=reduction)
    assert np.allclose(output_ms.asnumpy(),
                       expect.numpy(),
                       rtol=1e-4,
                       atol=1e-4,
                       equal_nan=False)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_triplet_margin_loss_float32(mode):
    """
    Feature: Input type of float32
    Description: Input type of [float32, float32, float32].
    Expectation: success.
    """
    context.set_context(mode=mode)
    data_type = np.float32
    anchor_array = np.array([[1.3, 20.5, 5.6],
                             [3.5, 4.8, 7.2],
                             [0.2, 0.01, 1],
                             [4, 4.1, 20]]).astype(data_type)
    positive_array = np.array([[2., 10., 1.],
                               [6., 7., 10.],
                               [13., 4., 1.],
                               [0.33, -4, -1.5]]).astype(data_type)
    negative_array = np.array([[2., 21., 6.],
                               [68., 9., 10.],
                               [131., 25., 16.],
                               [0.31, -0.14, -16.]]).astype(data_type)
    margin = np.float32(2.0)
    p = 1
    swap = False
    reduction = "none"
    eps = 1e-6

    anchor = Tensor(anchor_array)
    positive = Tensor(positive_array)
    negative = Tensor(negative_array)
    triplet_margin_loss = NetTripletMarginLoss(margin=margin, p=p, eps=eps,
                                               swap=swap, reduction=reduction)
    output_ms = triplet_margin_loss(anchor, positive, negative)

    torch_anchor = torch.tensor(anchor_array)
    torch_positive = torch.tensor(positive_array)
    torch_negative = torch.tensor(negative_array)
    expect = torch.nn.functional.triplet_margin_loss(torch_anchor, torch_positive,
                                                     torch_negative, margin=margin,
                                                     p=p, eps=eps, swap=swap,
                                                     reduction=reduction)
    assert np.allclose(output_ms.asnumpy(),
                       expect.numpy(),
                       rtol=1e-4,
                       atol=1e-4,
                       equal_nan=False)
