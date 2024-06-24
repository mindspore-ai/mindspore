from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops.operations.nn_ops as ops
import torch


class NetTripletMarginLoss(nn.Cell):

    def __init__(self, p=2, swap=False, eps=1e-6, reduction="mean"):
        super(NetTripletMarginLoss, self).__init__()
        self.triplet_margin_loss = ops.TripletMarginLoss(
            p=p, swap=swap, eps=eps, reduction=reduction)

    def construct(self, anchor, positive, negative, margin):
        return self.triplet_margin_loss(anchor, positive, negative, margin)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_triplet_margin_loss_float64():
    """
    Feature: Input type of float64
    Description: Input type of [float64, float64, float64, float32].
    Expectation: success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        data_type = np.float64
        anchor_array = np.array([[1.3, 20.5, 5.6]]).astype(data_type)
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
        mind_margin = Tensor(margin)
        triplet_margin_loss = NetTripletMarginLoss(p=p, swap=swap, reduction=reduction, eps=eps)
        output_ms = triplet_margin_loss(anchor, positive, negative, mind_margin)
        print(output_ms)
        torch_anchor = torch.tensor(anchor_array)
        torch_positive = torch.tensor(positive_array)
        torch_negative = torch.tensor(negative_array)
        torch_loss = torch.nn.TripletMarginLoss(margin=margin, p=p, swap=swap, reduction=reduction, eps=eps)
        expect = torch_loss(torch_anchor, torch_positive, torch_negative)
        assert np.allclose(output_ms.asnumpy(),
                           expect.numpy(),
                           rtol=1e-4,
                           atol=1e-4,
                           equal_nan=False)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_triplet_margin_loss_float32():
    """
    Feature: Input type of float32
    Description: Input type of [float32, float32, float32, float32].
    Expectation: success.
    """
    for mode in [context.GRAPH_MODE, context.PYNATIVE_MODE]:
        context.set_context(mode=mode, device_target="GPU")
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
        margin = 2.0
        p = 1
        swap = False
        reduction = "none"
        eps = 1e-6

        anchor = Tensor(anchor_array)
        positive = Tensor(positive_array)
        negative = Tensor(negative_array)
        mind_margin = Tensor(margin)
        triplet_margin_loss = NetTripletMarginLoss(p=p, swap=swap, reduction=reduction, eps=eps)
        output_ms = triplet_margin_loss(anchor, positive, negative, mind_margin)
        torch_anchor = torch.tensor(anchor_array)
        torch_positive = torch.tensor(positive_array)
        torch_negative = torch.tensor(negative_array)
        torch_loss = torch.nn.TripletMarginLoss(margin=margin, p=p, swap=swap, reduction=reduction, eps=eps)
        expect = torch_loss(torch_anchor, torch_positive, torch_negative)
        assert np.allclose(output_ms.asnumpy(),
                           expect.numpy(),
                           rtol=1e-4,
                           atol=1e-4,
                           equal_nan=False)
