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

import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class ReduceMin(nn.Cell):
    def __init__(self, keep_dims):
        super(ReduceMin, self).__init__()
        self.reduce_min = P.ReduceMin(keep_dims)

    def construct(self, x, axis):
        return self.reduce_min(x, axis)


def get_output(x, axis, keep_dims, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = ReduceMin(keep_dims)
    output = net(x, axis)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reduce_min():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    x0 = Tensor(np.random.normal(0, 1, [2, 3, 4, 4]).astype(np.float32))
    axis0 = 3
    keep_dims0 = True
    expect = get_output(x0, axis0, keep_dims0, False)
    output = get_output(x0, axis0, keep_dims0, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)

    x1 = Tensor(np.random.normal(0, 1, [2, 3, 4, 4]).astype(np.float32))
    axis1 = 3
    keep_dims1 = False
    expect = get_output(x1, axis1, keep_dims1, False)
    output = get_output(x1, axis1, keep_dims1, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)

    x2 = Tensor(np.random.normal(0, 1, [2, 3, 1, 4]).astype(np.float32))
    axis2 = 2
    keep_dims2 = True
    expect = get_output(x2, axis2, keep_dims2, False)
    output = get_output(x2, axis2, keep_dims2, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 0.0001, 0.0001)
