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
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, update_slots):
        super(Net, self).__init__()
        self.sparse_apply_adagrad_v2 = P.SparseApplyAdagradV2(lr=1e-8, update_slots=update_slots, \
                                        epsilon=1e-6)

    def construct(self, var, accum, grad, indices):
        out = self.sparse_apply_adagrad_v2(var, accum, grad, indices)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparseapplyadagradv2op_fp32():
    """
    Feature: SparseApplyAdagradV2 cpu op
    Description: Test output for fp32 dtype
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([[0.2, 0.1, 0.4, 0.7],
                                     [0.4, 0.5, 0.1, 0.8],
                                     [0.3, 0.4, 0.1, 0.9],
                                     [0.6, 0.8, 0.4, 0.5],
                                     [0.8, 0.9, 0.34, 0.56]]).astype(np.float32)), name="var")
    accum = Parameter(Tensor(np.array([[0.1, 0.2, 0.5, 0.9],
                                       [0.2, 0.4, 0.3, 0.1],
                                       [0.5, 0.6, 0.2, 0.6],
                                       [0.55, 0.16, 0.3, 0.8],
                                       [0.75, 0.16, 0.34, 0.18]]).astype(np.float32)), name="accum")
    gradient = Tensor(np.array([[0.5, 0.2, 0.6, 0.5],
                                [0.3, 0.5, 0.7, 0.4],
                                [0.5, 0.4, 0.3, 0.1],
                                [0.15, 0.3, 0.5, 0.3],
                                [0.52, 0.19, 0.37, 0.68]]).astype(np.float32))
    indices = Tensor([0, 1, 2, 3, 4], mindspore.int32)
    sparse_apply_adagrad_v2 = Net(update_slots=True)
    var_out, accum_out = sparse_apply_adagrad_v2(var, accum, gradient, indices)
    expect_var = np.array([[0.19999999, 0.09999999, 0.4, 0.7],
                           [0.4, 0.5, 0.09999999, 0.8],
                           [0.3, 0.4, 0.09999999, 0.9],
                           [0.6, 0.8, 0.4, 0.5],
                           [0.8, 0.9, 0.34, 0.56]]).astype(np.float32)
    expect_accum = np.array([[0.35, 0.24000001, 0.86, 1.15],
                             [0.29000002, 0.65, 0.78999996, 0.26000002],
                             [0.75, 0.76000005, 0.29000002, 0.61],
                             [0.5725, 0.25, 0.55, 0.89],
                             [1.0204, 0.1961, 0.4769, 0.6424]]).astype(np.float32)
    assert np.allclose(var_out.asnumpy(), expect_var, rtol=1e-3)
    assert np.allclose(accum_out.asnumpy(), expect_accum, rtol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparseapplyadagradv2op_update_slot_false():
    """
    Feature: SparseApplyAdagradV2 cpu op
    Description: Test output with update_slot set to False
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([0.2]).astype(np.float32)), name="var")
    accum = Parameter(Tensor(np.array([0.1]).astype(np.float32)), name="accum")
    gradient = Tensor(np.array([0.7]).astype(np.float32))
    indices = Tensor([0], mindspore.int32)
    sparse_apply_adagrad_v2 = Net(update_slots=False)
    var_out, accum_out = sparse_apply_adagrad_v2(var, accum, gradient, indices)
    expect_var = np.array([0.19999999]).astype(np.float32)
    expect_accum = np.array([0.1]).astype(np.float32)
    assert np.all(var_out.asnumpy() == expect_var)
    assert np.all(accum_out.asnumpy() == expect_accum)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparseapplyadagradv2_dtype_not_supported():
    """
    Feature: SparseApplyAdagradV2 cpu op
    Description: Test output for unsupported dtype
    Expectation: Raises TypeError
    """
    with pytest.raises(TypeError):
        var = Parameter(Tensor(np.array([0.2]).astype(np.float64)), name="var")
        accum = Parameter(Tensor(np.array([0.1]).astype(np.float64)), name="accum")
        gradient = Tensor(np.array([0.7]).astype(np.float64))
        indices = Tensor([0], mindspore.int32)
        sparse_apply_adagrad_v2 = Net(update_slots=True)
        sparse_apply_adagrad_v2(var, accum, gradient, indices)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparseapplyadagradv2_dynamic_shape_support():
    """
    Feature: SparseApplyAdagradV2 cpu op
    Description: Test support for dynamic shape
    Expectation: Op should accept dynamic input shapes
    """
    sparse_apply_adagrad_v2 = Net(update_slots=True)
    for i in range(2, 6):
        var = Parameter(Tensor(np.random.rand(i, i), mstype.float32), name="var")
        accum = Parameter(Tensor(np.random.rand(i, i), mstype.float32), name="accum")
        gradient = Parameter(Tensor(np.random.rand(i, i), mstype.float32), name="grad")
        indices = Tensor(np.arange(i), mstype.int32)

        sparse_apply_adagrad_v2(var, accum, gradient, indices)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_sparseapplyadagradopv2():
    """
    Feature: Vmap feature on SparseApplyAdagradv2 cpu op
    Description: Compare the vmap result with the manually batch result.
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([[0.2], [0.1]]).astype(np.float32)))
    accum = Parameter(Tensor(np.array([[0.1], [0.1]]).astype(np.float32)))
    gradient = Tensor(np.array([[0.7], [0.1]]).astype(np.float32))
    indices = Tensor([[0], [0]], mindspore.int32)
    sparse_apply_adagrad = Net(update_slots=True)
    return_vmap = F.vmap(sparse_apply_adagrad, in_axes=(0, 0, 0, 0))(var, accum, gradient, indices)

    expect_var = np.array([[0.19999999], [0.1]]).astype(np.float32)
    expect_accum = np.array([[0.59], [0.11]]).astype(np.float32)
    assert len(return_vmap) == 2
    assert np.all(return_vmap[0].asnumpy() == expect_var)
    assert np.all(return_vmap[1].asnumpy() == expect_accum)
