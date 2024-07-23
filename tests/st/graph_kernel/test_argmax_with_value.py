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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class ArgMaxWithValue(nn.Cell):
    def __init__(self, keep_dims, axis, index):
        super(ArgMaxWithValue, self).__init__()
        self.arg_max = P.ArgMaxWithValue(axis=axis, keep_dims=keep_dims)
        self.add = P.Add()
        self.ind = index

    def construct(self, x):
        res = self.arg_max(x)
        return self.add(res[self.ind], res[self.ind])


def get_output(x, keep_dims, axis, index, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = ArgMaxWithValue(keep_dims, axis, index)
    output = net(x)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_argmax_with_value():
    """
    Feature: graph_kernel_expander
    Description: test ArgMaxWithValue when enable_graph_kernel=True
    Expectation: the results are consistent whether using graph kernel or not
    """
    context.set_context(mode=context.GRAPH_MODE)
    x0 = Tensor(np.random.normal(0, 1, [2, 3, 4, 4]).astype(np.float32))
    axis0 = -1
    expect = get_output(x0, False, axis0, 0, False)
    output = get_output(x0, False, axis0, 0, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)

    x1 = Tensor(np.random.normal(0, 1, [2, 3, 1, 4]).astype(np.float32))
    axis1 = 1
    expect = get_output(x1, True, axis1, 1, False)
    output = get_output(x1, True, axis1, 1, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)

    x2 = Tensor(np.random.normal(0, 1, [2, 3, 1, 4]).astype(np.float16))
    axis2 = 0
    expect = get_output(x2, True, axis2, 1, False)
    output = get_output(x2, True, axis2, 1, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)

    x3 = Tensor(np.random.normal(0, 1, [2, 3, 1, 4]).astype(np.float32))
    axis3 = -1
    expect = get_output(x3, False, axis3, 1, False)
    output = get_output(x3, False, axis3, 1, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)
