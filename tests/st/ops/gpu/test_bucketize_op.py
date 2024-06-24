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

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations.math_ops import Bucketize
from mindspore.nn import Cell

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class BucketizeNet(Cell):

    def __init__(self, boundaries):
        super().__init__()
        self.bucketize = Bucketize(boundaries=boundaries)

    def construct(self, x):
        return self.bucketize(x)


def dyn_case():
    boundaries = [
        -1.743, -1.397, -0.702, -0.4631, -0.146, 0.0859, 0.814, 0.836, 1.704,
        2.392
    ]
    net = BucketizeNet(boundaries=boundaries)

    x_dyn = Tensor(shape=[None, None], dtype=ms.float64)
    net.set_inputs(x_dyn)

    x = Tensor(
        np.array([[0.42987306, 0.02847828, 0.59385591, 0.7040952, 0.27390435],
                  [0.32904094, 0.63063352, 0.70752448, 0.24763578, 0.99662956],
                  [0.66478424, 0.70580542, 0.92749155, 0.72736302, 0.24973136],
                  [0.79918445, 0.68613469, 0.9526593, 0.12412648,
                   0.15175918]]).astype(np.float64))
    out = net(x)

    expect_shape = (4, 5)
    assert out.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_bucketize_dyn():
    """
    Feature: test Bucketize ops in gpu.
    Description: Test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    dyn_case()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bucketize_4x5_float64():
    """
    Feature: Bucketize
    Description: Test int32 of input
    Expectation: The results are as expected
    """
    x_np = np.array(
        [[0.42987306, 0.02847828, 0.59385591, 0.7040952, 0.27390435],
         [0.32904094, 0.63063352, 0.70752448, 0.24763578, 0.99662956],
         [0.66478424, 0.70580542, 0.92749155, 0.72736302, 0.24973136],
         [0.79918445, 0.68613469, 0.9526593, 0.12412648,
          0.15175918]]).astype(np.float64)
    boundaries = [
        -1.743, -1.397, -0.702, -0.4631, -0.146, 0.0859, 0.814, 0.836, 1.704,
        2.392
    ]
    net = BucketizeNet(boundaries)
    output_ms = net(Tensor(x_np))
    expect_output = np.array([[6, 5, 6, 6, 6], [6, 6, 6, 6, 8],
                              [6, 6, 8, 6, 6], [6, 6, 8, 6, 6]])
    assert np.allclose(output_ms.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bucketize_4x5x6_int32():
    """
    Feature: Bucketize
    Description: Test int32 of input
    Expectation: The results are as expected
    """
    x_np = np.array([[[0, -1, 0], [-1, 0, 0]]]).astype(np.int32)
    boundaries = [
        -1.743, -1.397, -0.702, -0.4631, -0.146, 0.0859, 0.814, 0.836, 1.704,
        2.392
    ]
    net = BucketizeNet(boundaries)
    output_ms = net(Tensor(x_np))
    expect_output = np.array([[[5, 2, 5], [2, 5, 5]]])
    assert np.allclose(output_ms.asnumpy(), expect_output)
