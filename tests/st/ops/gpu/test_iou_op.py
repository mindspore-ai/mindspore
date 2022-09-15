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
# ============================================================================

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetIOU(nn.Cell):
    def __init__(self, mode):
        super(NetIOU, self).__init__()
        self.encode = P.IOU(mode=mode)

    def construct(self, anchor, groundtruth):
        return self.encode(anchor, groundtruth)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_iou_gpu():
    """
    Feature: test iou op in gpu.
    Description: test iou in gpu.
    Expectation: The output is equal to the expected value.
    """
    pos1 = [101, 169, 246, 429]
    pos2 = [121, 138, 304, 374]
    mode = "iou"
    pos1_box = Tensor(np.array(pos1).reshape(1, 4), mindspore.float32)
    pos2_box = Tensor(np.array(pos2).reshape(1, 4), mindspore.float32)
    expect_result = np.array(0.46551168, np.float32)

    error = np.ones(shape=[1]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    overlaps = NetIOU(mode)
    output = overlaps(pos1_box, pos2_box)
    diff = output.asnumpy() - expect_result
    assert np.all(abs(diff) < error)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    overlaps = NetIOU(mode)
    output = overlaps(pos1_box, pos2_box)
    diff = output.asnumpy() - expect_result
    assert np.all(abs(diff) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_iou_gpu_dynamic_shape():
    """
    Feature: test iou op in gpu.
    Description: test iou in dynamic shape.
    Expectation: The output is equal to the expected value.
    """
    pos1 = [101, 169, 246, 429]
    pos2 = [121, 138, 304, 374]
    mode = "iou"
    pos1_box = Tensor(np.array(pos1).reshape(1, 4), mindspore.float32)
    pos2_box = Tensor(np.array(pos2).reshape(1, 4), mindspore.float32)
    expect_result = np.array(0.46551168, np.float32)

    error = np.ones(shape=[1]) * 1.0e-6

    pos1_dyn = Tensor(shape=[None, 4], dtype=mindspore.float32)
    pos2_dyn = Tensor(shape=[None, 4], dtype=mindspore.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    dynamic_overlaps = NetIOU(mode)
    dynamic_overlaps.set_inputs(pos1_dyn, pos2_dyn)
    output = dynamic_overlaps(pos1_box, pos2_box)
    diff = output.asnumpy() - expect_result
    assert np.all(abs(diff) < error)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    dynamic_overlaps = NetIOU(mode)
    dynamic_overlaps.set_inputs(pos1_dyn, pos2_dyn)
    output = dynamic_overlaps(pos1_box, pos2_box)
    diff = output.asnumpy() - expect_result
    assert np.all(abs(diff) < error)
