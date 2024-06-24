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
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
import mindspore.ops.operations.nn_ops as P
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self, use_locking=False):
        super(Net, self).__init__()
        self.sparse_apply_adagrad_da = P.SparseApplyAdagradDA(use_locking)

    def construct(self, *x):
        return self.sparse_apply_adagrad_da(*x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparseapplyadagradda_fp32():
    """
    Feature: SparseApplyAdagradDA gpu op
    Description: Test output for fp32 dtype
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([[1, 2], [1, 2]]).astype(np.float32)), name="var")
    grad_accum = Parameter(Tensor(np.array([[2, 1], [3, 1]]).astype(np.float32)), name="grad_accum")
    grad_square_accum = Parameter(Tensor(np.array([[4, 1], [5, 1]]).astype(np.float32)), name="grad_square_accum")
    grad = Tensor(np.array([[5, 1], [6, 1]]).astype(np.float32))
    indices = Tensor(np.array([1, 1], dtype=np.int32))
    lr = Tensor(2, mstype.float32)
    l1 = Tensor(-1, mstype.float32)
    l2 = Tensor(1, mstype.float32)
    global_step = Tensor(1, mstype.int64)
    net = Net()
    var_out = net(var, grad_accum, grad_square_accum,
                  grad, indices, lr, l1, l2, global_step)
    expect_var = np.array([[1., 2.],
                           [-2.7656946, -1.6076951]]).astype(np.float32)

    assert np.all(var_out.asnumpy() == expect_var)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparseapplyadagradda_fp16():
    """
    Feature: SparseApplyAdagradDA gpu op
    Description: Test output for fp16 dtype
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([[1, 2], [1, 2]]).astype(np.float16)), name="var")
    grad_accum = Parameter(Tensor(np.array([[2, 1], [3, 1]]).astype(np.float16)), name="grad_accum")
    grad_square_accum = Parameter(Tensor(np.array([[4, 1], [5, 1]]).astype(np.float16)), name="grad_square_accum")
    grad = Tensor(np.array([[5, 1], [6, 1]]).astype(np.float16))
    indices = Tensor(np.array([1, 1], dtype=np.int32))
    lr = Tensor(2, mstype.float16)
    l1 = Tensor(-1, mstype.float16)
    l2 = Tensor(1, mstype.float16)
    global_step = Tensor(1, mstype.int64)
    net = Net()
    var_out = net(var, grad_accum, grad_square_accum,
                  grad, indices, lr, l1, l2, global_step)
    expect_var = np.array([[1., 2.],
                           [-2.7656946, -1.6076951]]).astype(np.float16)

    assert np.all(var_out.asnumpy() == expect_var)
