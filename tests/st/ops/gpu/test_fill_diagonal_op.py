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
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function


class FillDiagonalNet(nn.Cell):
    def __init__(self, nptype, fill_value, wrap):
        super(FillDiagonalNet, self).__init__()
        self.fill_diagonal = P.FillDiagonal(fill_value, wrap)
        self.a_np = np.zeros((7, 3)).astype(nptype)
        self.a = Tensor(self.a_np)
        self.expect = np.array([[5., 0., 0.],
                                [0., 5., 0.],
                                [0., 0., 5.],
                                [0., 0., 0.],
                                [5., 0., 0.],
                                [0., 5., 0.],
                                [0., 0., 5.]]).astype(nptype)


    @ms_function
    def construct(self):
        return self.fill_diagonal(self.a)



def fill_diagonal(nptype, fill_value, wrap):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    fill_diagonal_ = FillDiagonalNet(nptype, fill_value, wrap)
    fill_diagonal_output = fill_diagonal_().asnumpy()
    fill_diagonal_expect = fill_diagonal_.expect
    assert np.allclose(fill_diagonal_output, fill_diagonal_expect)


def fill_diagonal_pynative(nptype, fill_value, wrap):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    fill_diagonal_ = FillDiagonalNet(nptype, fill_value, wrap)
    fill_diagonal_output = fill_diagonal_().asnumpy()
    fill_diagonal_expect = fill_diagonal_.expect
    assert np.allclose(fill_diagonal_output, fill_diagonal_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fill_diagonal_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for FillDiagonal
    Expectation: the result match to numpy
    """
    fill_diagonal(np.float32, 5., True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fill_diagonal_pynative_int32():
    """
    Feature: ALL To ALL
    Description: test cases for FillDiagonal
    Expectation: the result match to numpy
    """
    fill_diagonal_pynative(np.int32, 5., True)
