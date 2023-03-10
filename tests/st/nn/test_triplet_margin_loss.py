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
import torch
import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    ms_margin = Tensor(margin)
    triplet_margin_loss = nn.TripletMarginLoss(p=p, eps=eps, swap=swap, reduction=reduction, margin=ms_margin)
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
    ms_margin = Tensor(margin)
    triplet_margin_loss = nn.TripletMarginLoss(p=p, eps=eps, swap=swap, reduction=reduction, margin=ms_margin)
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
