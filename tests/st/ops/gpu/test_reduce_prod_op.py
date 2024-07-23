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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class ReduceProd(nn.Cell):
    def __init__(self, keep_dims):
        super(ReduceProd, self).__init__()
        self.reduce_prod = P.ReduceProd(keep_dims=keep_dims)

    def construct(self, x, axis):
        return self.reduce_prod(x, axis)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('decimal, dtype',
                         [(1e-10, np.int8), (1e-3, np.float16), (1e-5, np.float32), (1e-8, np.float64)])
@pytest.mark.parametrize('shape, axis, keep_dims',
                         [((2, 3, 4, 4), 3, True), ((2, 3, 4, 4), 3, False), ((2, 3, 1, 4), 2, True),
                          ((2, 3, 1, 4), 2, False), ((2, 3, 4, 4), None, True), ((2, 3, 4, 4), None, False),
                          ((2, 3, 4, 4), -2, False), ((2, 3, 4, 4), (-2, -1), False), ((1, 1, 1, 1), None, True)])
def test_reduce_prod(decimal, dtype, shape, axis, keep_dims):
    """
    Feature: ALL To ALL
    Description: test cases for ReduceProd
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = np.random.rand(*shape).astype(dtype)
    tensor_x = Tensor(x)

    reduce_prod = ReduceProd(keep_dims)
    ms_axis = axis if axis is not None else ()
    output = reduce_prod(tensor_x, ms_axis)

    expect = np.prod(x, axis=axis, keepdims=keep_dims)
    diff = abs(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * decimal
    assert np.all(diff < error)
    assert output.shape == expect.shape
