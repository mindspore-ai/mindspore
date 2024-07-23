# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore import dtype as mstype

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d_grad = G.Conv2DBackpropFilter(4, 1)
        yt = Tensor(
            np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32))
        self.y = Parameter(yt, name='y')
        self.get_shape = P.Shape()

    def construct(self, x_, out_):
        return self.conv2d_grad(out_, x_, self.get_shape(self.y))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net():
    """
    Feature: Test tbe kernel select need reduce and raise precision.
    Description: Input is dynamic shape.
    Expectation: Output is close to the expect_out.
    """
    x = Tensor(np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float16))
    out = Tensor(np.array([[[
        [-5, -4, 0, 8],
        [-10, -2, 2, 3],
        [0, -2, -4, -7],
        [-3, -2, -3, -16]]]]).astype(np.float16))
    operator = Net()
    operator.set_inputs(Tensor(shape=[None, 1, 6, 6], dtype=mstype.float16),
                        Tensor(shape=[None, 1, 4, 4], dtype=mstype.float16))
    output = operator(x, out)
    expect_out = np.array(
        [[[[-60., -142., -265.], [-104., -211., -322.], [-102., -144., -248.]]]]).astype(np.float16)
    assert np.allclose(output.asnumpy(), expect_out, 1e-3, 1e-3)
