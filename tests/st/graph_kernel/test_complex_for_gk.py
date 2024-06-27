# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
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


class ComplexNet(nn.Cell):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.mul = P.Mul()
        self.add = P.Add()
        self.sub = P.Sub()
        self.div = P.Div()

    def construct(self, x, y):
        mul_xy = self.mul(x, y)
        add_xy = self.add(x, y)
        sub_xy = self.sub(mul_xy, add_xy)
        return self.div(mul_xy, sub_xy)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_complex():
    """
    Feature: easy complex test case for graph_kernel on gpu.
    Description: gpu test case, use graph_kernel expand complex ops.
    Expectation: the result is equal to numpy result.
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    x = np.array([[3 + 4j, 5 - 2j, 6 + 8j], [3 + 4j, 5 - 2j, 6 + 8j]]).astype(np.complex64)
    y = np.array([3, 4, 5]).astype(np.float32)
    net = ComplexNet()
    output = net(Tensor(x), Tensor(y))
    expect_np = (x * y) / (x * y - (x + y))
    assert np.allclose(expect_np, output.asnumpy(), 0.0001, 0.0001)
