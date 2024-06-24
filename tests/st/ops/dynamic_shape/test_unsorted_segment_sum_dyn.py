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
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class UnsortedSegmentSumDynamicShapeNetMS(nn.Cell):
    def __init__(self):
        super().__init__()
        self.uss = P.UnsortedSegmentSum()

    def construct(self, x, segment_ids, num_segments):
        return self.uss(x, segment_ids, num_segments)


class UnsortedSegmentSumDynamicShapeNetMSBeta(nn.Cell):
    def __init__(self, numsegments):
        super().__init__()
        self.uss = P.UnsortedSegmentSum()
        self.numsegments = numsegments

    def construct(self, x, segment_ids):
        return self.uss(x, segment_ids, self.numsegments)


def dyn_case():
    x = np.arange(1, 10).reshape(3, 3).astype(np.float32)
    input_x_dyn = Tensor(shape=[None, 3], dtype=mindspore.float32)
    input_x = Tensor(x)
    segment_ids = Tensor([0, 1, 0], mindspore.int32)
    num_segments = Tensor([2,], mindspore.int32)
    expect_np = np.array([[8, 10, 12], [4, 5, 6]], dtype=np.float32)
    net = UnsortedSegmentSumDynamicShapeNetMS()
    net.set_inputs(input_x_dyn, segment_ids, num_segments)
    output = net(input_x, segment_ids, num_segments)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


def dyn_case_beta():
    x = np.arange(1, 10).reshape(3, 3).astype(np.float32)
    input_x_dyn = Tensor(shape=[3, None], dtype=mindspore.float32)
    input_x = Tensor(x)
    segment_ids = Tensor([0, 1, 0], mindspore.int32)
    num_segments = 2
    expect_np = np.array([[8, 10, 12], [4, 5, 6]], dtype=np.float32)
    net = UnsortedSegmentSumDynamicShapeNetMSBeta(num_segments)
    net.set_inputs(input_x_dyn, segment_ids)
    output = net(input_x, segment_ids)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uss_dyn_cpu():
    """
    Feature: test UnsortedSegmentSum dynamic shape on CPU, all inputs are tensor.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_case()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_uss_dyn_gpu():
    """
    Feature: test UnsortedSegmentSum dynamic shape on GPU, all inputs are tensor.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_uss_dyn_cpu_beta():
    """
    Feature: test UnsortedSegmentSum dynamic shape on CPU, num_segments is a var.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_case_beta()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_case_beta()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_uss_dyn_gpu_beta():
    """
    Feature: test UnsortedSegmentSum dynamic shape on GPU, num_segments is a var.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case_beta()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case_beta()
