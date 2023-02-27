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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.api import jit

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class NetReduceStd(nn.Cell):
    def __init__(self, axis, keep_dims):
        super(NetReduceStd, self).__init__()
        self._axis = axis
        self._keep_dims = keep_dims

    @jit
    def construct(self, indice):
        if self._axis is None:
            return F.std_mean(indice, ddof=False, keepdims=self._keep_dims)
        return F.std_mean(indice, axis=self._axis, ddof=False, keepdims=self._keep_dims)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('axis', [None, 0, 1, -1, (-1, 0, 1), (0, 1, 2)])
@pytest.mark.parametrize('keep_dims', [True, False])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_reduce_std_op(axis, keep_dims, dtype):
    """
    Feature: Reduce std CPU operation
    Description: input the unbiased and keep_dims, test the output value
    Expectation: the reduce std result match to numpy
    """
    reduce_std = NetReduceStd(axis, keep_dims)
    tensor_x = Tensor(np.array([
        [[0., 2., 1., 4., 0., 2.], [3., 1., 2., 2., 4., 0.]],
        [[2., 0., 1., 5., 0., 1.], [1., 0., 0., 4., 4., 3.]],
        [[4., 1., 4., 0., 0., 0.], [2., 5., 1., 0., 1., 3.]]
    ]).astype(dtype))
    output = reduce_std(tensor_x)

    expect_std0 = np.std(tensor_x.asnumpy(), axis=axis, keepdims=keep_dims, dtype=dtype)
    expect_mean0 = np.mean(tensor_x.asnumpy(), axis=axis, keepdims=keep_dims, dtype=dtype)
    np.allclose(output[0].asnumpy(), expect_std0, 0.0001, 0.0001)
    np.allclose(output[1].asnumpy(), expect_mean0, 0.0001, 0.0001)


class ReduceStdDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(ReduceStdDynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, x):
        x_unique, _ = self.unique(x)
        x_unique = self.reshape(x_unique, (2, 5))
        return F.std_mean(x_unique, ddof=False, keepdims=False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reduce_std_dynamic_shape(mode):
    """
    Feature: test ReduceStd dynamic_shape feature.
    Description: test ReduceStd dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([8., -3., 0., 0., 10., 1., 21., -3., 10., 8., 3, 4, 5, 6]).astype(np.float32))
    output = ReduceStdDynamicShapeNet()(x)

    np_x = np.array([8., -3., 0., 10, 1., 21., 3, 4, 5, 6], dtype=np.float32)
    expect_output_std = np.std(np_x, keepdims=False)
    expect_output_mean = np.mean(np_x, keepdims=False)

    assert (output[0].asnumpy() == expect_output_std).all()
    assert (output[1].asnumpy() == expect_output_mean).all()
