# Copyright 2021 Huawei Technologies Co., Ltd
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
from typing import Generic
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register
import scipy as scp
import numpy as np
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class LU(PrimitiveWithInfer):
    """
    LU decomposition with partial pivoting
    P.A = L.U
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LU")
        self.init_prim_io_names(inputs=['x'], outputs=['lu', 'pivots', 'permutation'])

    def __infer__(self, x):
        x_shape = list(x['shape'])
        x_dtype = x['dtype']
        pivots_shape = []
        permutation_shape = []
        ndim = len(x_shape)
        if ndim == 0:
            pivots_shape = x_shape
            permutation_shape = x_shape
        elif ndim == 1:
            pivots_shape = x_shape[:-1]
            # permutation_shape = x_shape[:-1]
        else:
            pivots_shape = x_shape[-2:-1]
            # permutation_shape = x_shape[-2:-1]

        output = {
            'shape': (x_shape, pivots_shape, permutation_shape),
            'dtype': (x_dtype, mstype.int32, mstype.int32),
            'value': None
        }
        return output


class LuNet(nn.Cell):
    def __init__(self):
        super(LuNet, self).__init__()
        self.lu = LU()

    def construct(self, a):
        return self.lu(a)


@pytest.mark.platform_x86_gpu
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_lu_net(n: int, dtype: Generic):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    expect, _ = scp.linalg.lu_factor(a)
    mscp_lu_net = LuNet()
    # mindspore tensor is row major but gpu cusolver is col major, so we should transpose it.
    tensor_a = Tensor(a)
    tensor_a = mnp.transpose(tensor_a)
    output, _, _ = mscp_lu_net(tensor_a)
    # mindspore tensor is row major but gpu cusolver is col major, so we should transpose it.
    output = mnp.transpose(output)
    rtol = 1.e-4
    atol = 1.e-5
    assert np.allclose(expect, output.asnumpy(), rtol=rtol, atol=atol)
