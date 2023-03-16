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

import pytest
import numpy as np
from mindspore import Tensor
import mindspore.ops.operations.math_ops as op
from mindspore.nn import Cell
import mindspore.context as context
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class Quantile(Cell):
    def __init__(self, dim=0, keep_dims=False):
        super().__init__()
        self.quantile = op.Quantile(dim=dim, keep_dims=keep_dims)

    def construct(self, x, q):
        return self.quantile(x, q)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="I6NEX6, I6NEJF")
def test_quantile_fp32():
    """
    Feature: Quantile
    Description: Test of input
    Expectation: The results are as expected
    """
    type_i = np.float32
    ertol_loss = 1e-04
    x = np.array([[1.0, 5.0, 9.0, 13], [2, 6, 10, 14],
                  [3, 7, 11, 15], [4, 8, 12, 16]]).astype(type_i)
    q = np.array([0.25, 0.5, 0.75]).astype(type_i)
    dim = 0
    keep_dims = True
    net = Quantile(dim=dim, keep_dims=keep_dims)
    output = net(Tensor(x), Tensor(q))
    output = output.asnumpy()
    expect_output = np.array([[[1.75, 5.75, 9.75, 13.75]], [[2.5, 6.5, 10.5, 14.5]],
                              [[3.25, 7.25, 11.25, 15.25]]]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="I6NEX6, I6NEJF")
def test_quantile_fp64():
    """
    Feature: Quantile
    Description: Test of input
    Expectation: The results are as expected
    """
    type_i = np.float64
    ertol_loss = 1e-05
    x = np.array([[1.0, 5.0, 9.0, 13], [2, 6, 10, 14],
                  [3, 7, 11, 15], [4, 8, 12, 16]]).astype(type_i)
    q = np.array([0.25, 0.5, 0.75]).astype(type_i)
    dim = 0
    keep_dims = True
    net = Quantile(dim=dim, keep_dims=keep_dims)
    output = net(Tensor(x), Tensor(q))
    output = output.asnumpy()
    expect_output = np.array([[[1.75, 5.75, 9.75, 13.75]], [[2.5, 6.5, 10.5, 14.5]],
                              [[3.25, 7.25, 11.25, 15.25]]]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)
