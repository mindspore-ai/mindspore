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
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

class Einsum(nn.Cell):
    def __init__(self, equation):
        super().__init__()
        self.einsum = P.Einsum(equation)

    def construct(self, *inputs):
        out = self.einsum(inputs)
        return out

class EinsumGrad(nn.Cell):
    def __init__(self, equation):
        super().__init__()
        self.einsum_grad = G.EinsumGrad(equation)

    def construct(self, *inputs):
        num = len(inputs)
        inp_data = inputs[0:num - 1]
        dout = inputs[num - 1 : num]
        dx = self.einsum_grad(inp_data, dout)
        return dx

def einsum_test_cases(nptype, loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    test_cases = [["abcd->dacb", [[2, 3, 1, 1]]],
                  ["ijk->ik", [[1, 2, 3]]],
                  ["ij,ij->ij", [[2, 3], [2, 3]]],
                  ["ij,kl->ijkl", [[3, 2], [2, 3]]],
                  ["ij,jk->ik", [[3, 2], [2, 3]]]
                 ]
    for cur_case in test_cases:
        equation = cur_case[0]
        shapes = cur_case[1]
        ms_data = []
        np_data = []
        for cur_shape in shapes:
            cur_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(cur_shape).astype(np.float64)
            ms_data.append(Tensor(cur_data.astype(nptype)))
            np_data.append(cur_data)
        net = Einsum(equation)
        ms_out = net(*ms_data)
        np_out = np.einsum(equation, *np_data)
        assert np.allclose(ms_out.asnumpy(), np_out.astype(nptype), loss, loss)
        grad_net = EinsumGrad(equation)
        ms_dx = grad_net(*ms_data, Tensor(np_out.astype(nptype)))
        print(ms_dx)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_einsum_graph_float16():
    """
    Feature: test transpose/ reduce_sum/dot/mul/transpose_with_ell/batchmatmul
    Description: test the accuracy and precision of the preceding test cases in float16 types
    Expectation: the diff between the result and the operator of np.einsum is within the loss range
    """
    einsum_test_cases(np.float16, 1e-3)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_einsum_graph_float32():
    """
    Feature: test transpose/ reduce_sum/dot/mul/transpose_with_ell/batchmatmul
    Description: test the accuracy and precision of the preceding test cases in float32 types
    Expectation: the diff between the result and the operator of np.einsum is within the loss range
    """
    einsum_test_cases(np.float32, 1e-4)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_einsum_graph_float64():
    """
    Feature: test transpose/ reduce_sum/dot/mul/transpose_with_ell/batchmatmul
    Description: test the accuracy and precision of the preceding test cases in float64 types
    Expectation: the diff between the result and the operator of np.einsum is within the loss range
    """
    einsum_test_cases(np.float64, 1e-5)
