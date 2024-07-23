# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import composite as C


class SparseDenseMatmulNet(nn.Cell):

    def __init__(self, adjoint_st=False, adjoint_dt=False):
        super(SparseDenseMatmulNet, self).__init__()
        self.matmul = nn.SparseTensorDenseMatmul(adjoint_st, adjoint_dt)

    def construct(self, indices, values, dens_shape, dense):
        return self.matmul(indices, values, dens_shape, dense)


class GradNet(nn.Cell):

    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, indices, values, dens_shape, dense):
        return self.grad(self.network)(indices, values, dens_shape, dense)


def judge_result_correct(result, expect):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sparse_tensor_dense_mul(context_mode):
    """
    Feature: SparseTensorDenseMul op.
    Description: test SparseTensorDenseMul
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context_mode, device_target='Ascend')
    net = SparseDenseMatmulNet()

    x1_shape = Tensor([3, 4], dtype=ms.int64)
    x1_indices = Tensor([[0, 1], [1, 2]], dtype=ms.int64)
    x1_values = Tensor([1, 2], dtype=ms.float32)
    x2 = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ms.float32)
    out = net(x1_indices, x1_values, x1_shape, x2)

    expected = np.array([[2, 2], [6, 6], [0, 0]], np.float32)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-3)
