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
import mindspore.context as context
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
import mindspore.ops.operations.nn_ops as P


class SparseApplyCenteredRMSPropNet(nn.Cell):
    def __init__(self, use_locking=False):
        super(SparseApplyCenteredRMSPropNet, self).__init__()
        self.sparse_apply_centered_rms_prop = P.SparseApplyCenteredRMSProp(use_locking=False)

    def construct(self, var, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices):
        out = self.sparse_apply_centered_rms_prop(var, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_apply_centered_rms_prop_graph_1():
    """
    Feature: Test whether the output of Var calculated by mindspore and tensorflow are equal.
    Description: Inputs are Tensors in shape [2, 2]for mutable tensors, value for scalar and shape [2] for indices.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)), name="var")
    mg = Parameter(Tensor(np.array([[0.1, 0.3], [0.1, 0.5]]).astype(np.float32)), name="mg")
    ms = Parameter(Tensor(np.array([[0.2, 0.1], [0.1, 0.2]]).astype(np.float32)), name="ms")
    mom = Parameter(Tensor(np.array([[0.2, 0.1], [0.1, 0.2]]).astype(np.float32)), name="mom")
    lr = Tensor(0.001, mstype.float32)
    rho = Tensor(1e-10, mstype.float32)
    momentum = Tensor(0.001, mstype.float32)
    epsilon = Tensor(0.01, mstype.float32)
    grad = Parameter(Tensor(np.array([[0.3, 0.4], [0.1, 0.2]]).astype(np.float32)))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    sparse_apply_centered_rms_prop_net = SparseApplyCenteredRMSPropNet(use_locking=False)
    sparse_apply_centered_rms_prop_output = sparse_apply_centered_rms_prop_net(var, mg, ms, mom, lr, rho, \
        momentum, epsilon, grad, indices)
    sparse_apply_centered_rms_prop_expected_output = np.array([[0.5968, 0.3959], [0.0989, 0.4978]]).astype(np.float32)

    print(sparse_apply_centered_rms_prop_output)
    print(sparse_apply_centered_rms_prop_expected_output)
    assert np.allclose(sparse_apply_centered_rms_prop_output.asnumpy(), \
        sparse_apply_centered_rms_prop_expected_output, rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_apply_centered_rms_prop_graph_2():
    """
    Feature: Test whether the output of Var calculated by mindspore and tensorflow are equal.
    Description: Inputs are Tensors in shape [2, 2]for mutable tensors, value for scalar and shape [2] for indices.
    Expectation: Success.
    """
    var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)), name="var")
    mg = Parameter(Tensor(np.array([[0.1, 0.3], [0.1, 0.5]]).astype(np.float32)), name="mg")
    ms = Parameter(Tensor(np.array([[0.2, 0.1], [0.1, 0.2]]).astype(np.float32)), name="ms")
    mom = Parameter(Tensor(np.array([[0.2, 0.1], [0.1, 0.2]]).astype(np.float32)), name="mom")
    lr = Tensor(0.001, mstype.float32)
    rho = Tensor(1e-10, mstype.float32)
    momentum = Tensor(0.001, mstype.float32)
    epsilon = Tensor(0.01, mstype.float32)
    grad = Parameter(Tensor(np.array([[0.3, 0.4], [0.1, 0.2]]).astype(np.float32)))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    sparse_apply_centered_rms_prop_net = SparseApplyCenteredRMSPropNet(use_locking=False)
    sparse_apply_centered_rms_prop_output = sparse_apply_centered_rms_prop_net(var, mg, ms, mom, lr, rho, \
        momentum, epsilon, grad, indices)
    sparse_apply_centered_rms_prop_expected_output = np.array([[0.5968, 0.3959], [0.0989, 0.4978]]).astype(np.float32)

    print(sparse_apply_centered_rms_prop_output)
    print(sparse_apply_centered_rms_prop_expected_output)
    assert np.allclose(sparse_apply_centered_rms_prop_output.asnumpy(), \
        sparse_apply_centered_rms_prop_expected_output, rtol=1e-3)
