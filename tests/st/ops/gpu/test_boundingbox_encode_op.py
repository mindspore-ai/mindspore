# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class NetBoundingBoxEncode(nn.Cell):
    def __init__(self, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
        super(NetBoundingBoxEncode, self).__init__()
        self.encode = P.BoundingBoxEncode(means=means, stds=stds)

    def construct(self, anchor, groundtruth):
        return self.encode(anchor, groundtruth)


def bbox2delta(proposals, gt, means, stds):
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)
    means = np.array(means, np.float32)
    stds = np.array(stds, np.float32)
    deltas = np.stack([(dx - means[0]) / stds[0], (dy - means[1]) / stds[1],
                       (dw - means[2]) / stds[2], (dh - means[3]) / stds[3]], axis=-1)

    return deltas


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_boundingbox_encode():
    anchor = np.array([[4, 1, 6, 9], [2, 5, 5, 9]]).astype(np.float32)
    gt = np.array([[3, 2, 7, 7], [1, 5, 5, 8]]).astype(np.float32)
    means = (0.1, 0.1, 0.2, 0.2)
    stds = (2.0, 2.0, 3.0, 3.0)
    anchor_box = Tensor(anchor, mindspore.float32)
    groundtruth_box = Tensor(gt, mindspore.float32)
    expect_deltas = bbox2delta(anchor, gt, means, stds)

    error = np.ones(shape=[2, 4]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    boundingbox_encode = NetBoundingBoxEncode(means, stds)
    output = boundingbox_encode(anchor_box, groundtruth_box)
    diff = output.asnumpy() - expect_deltas
    assert np.all(abs(diff) < error)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    boundingbox_encode = NetBoundingBoxEncode(means, stds)
    output = boundingbox_encode(anchor_box, groundtruth_box)
    diff = output.asnumpy() - expect_deltas
    assert np.all(abs(diff) < error)


def test_bounding_box_encode_functional():
    """
    Feature: test bounding_box_encode functional API.
    Description: test case for bounding_box_encode functional API.
    Expectation: the result match with expected result.
    """
    anchor_box = Tensor([[2, 2, 2, 3], [2, 2, 2, 3]], mstype.float32)
    groundtruth_box = Tensor([[1, 2, 1, 4], [1, 2, 1, 4]], mstype.float32)
    output = F.bounding_box_encode(anchor_box, groundtruth_box, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
    expected = np.array([[-1., 0.25, 0., 0.40551758], [-1., 0.25, 0., 0.40551758]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bounding_box_encode_functional_modes():
    """
    Feature: test bounding_box_encode functional API in PyNative and Graph modes.
    Description: test case for bounding_box_encode functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_bounding_box_encode_functional()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_bounding_box_encode_functional()
