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

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, update_slots):
        super(Net, self).__init__()
        self.sparse_apply_adagrad = P.SparseApplyAdagrad(lr=1e-8, update_slots=update_slots)

    def construct(self, var, accum, grad, indices):
        out = self.sparse_apply_adagrad(var, accum, grad, indices)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparseapplyadagradop_fp32():
    """
    Feature: SparseApplyAdagrad cpu op
    Description: Test output for fp32 dtype
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([0.2]).astype(np.float32)), name="var")
    accum = Parameter(Tensor(np.array([0.1]).astype(np.float32)), name="accum")
    gradient = Tensor(np.array([0.7]).astype(np.float32))
    indices = Tensor([0], mindspore.int32)
    sparse_apply_adagrad = Net(update_slots=True)
    var_out, accum_out = sparse_apply_adagrad(var, accum, gradient, indices)
    expect_var = np.array([0.19999999]).astype(np.float32)
    expect_accum = np.array([0.59]).astype(np.float32)
    assert np.all(var_out.asnumpy() == expect_var)
    assert np.all(accum_out.asnumpy() == expect_accum)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparseapplyadagradop_update_slot_false():
    """
    Feature: SparseApplyAdagrad cpu op
    Description: Test output with update_slot set to False
    Expectation: Output matching expected values
    """
    var = Parameter(Tensor(np.array([0.2]).astype(np.float32)), name="var")
    accum = Parameter(Tensor(np.array([0.1]).astype(np.float32)), name="accum")
    gradient = Tensor(np.array([0.7]).astype(np.float32))
    indices = Tensor([0], mindspore.int32)
    sparse_apply_adagrad = Net(update_slots=False)
    var_out, accum_out = sparse_apply_adagrad(var, accum, gradient, indices)
    print(var_out, accum_out)
    expect_var = np.array([0.19999999]).astype(np.float32)
    expect_accum = np.array([0.1]).astype(np.float32)
    assert np.all(var_out.asnumpy() == expect_var)
    assert np.all(accum_out.asnumpy() == expect_accum)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparseapplyadagrad_dtype_not_supported():
    """
    Feature: SparseApplyAdagrad cpu op
    Description: Test output for unsupported dtype
    Expectation: Raises TypeError
    """
    with pytest.raises(TypeError):
        var = Parameter(Tensor(np.array([0.2]).astype(np.float64)), name="var")
        accum = Parameter(Tensor(np.array([0.1]).astype(np.float64)), name="accum")
        gradient = Tensor(np.array([0.7]).astype(np.float64))
        indices = Tensor([0], mindspore.int32)
        sparse_apply_adagrad = Net(update_slots=True)
        sparse_apply_adagrad(var, accum, gradient, indices)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_sparseapplyadagradop():
    """
    Feature: Vmap feature on SparseApplyAdagrad cpu op
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
