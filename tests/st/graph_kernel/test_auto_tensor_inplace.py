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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P


class ElemwiseNet(Cell):
    def __init__(self):
        super(ElemwiseNet, self).__init__()
        self.add = P.Add()
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.matmul = P.MatMul()

    def construct(self, x, y, z):
        res_1 = self.matmul(x, y)
        res_2 = self.sub(res_1, z)
        res_3 = self.exp(res_2)
        res_4 = self.add(res_3, z)
        return res_4


def fusion_net_get_output(x, y, z, enable_auto_tensor_inplace=False):
    context.set_context(enable_graph_kernel=True)
    if enable_auto_tensor_inplace:
        context.set_context(graph_kernel_flags="--enable_auto_tensor_inplace=true")
    net = ElemwiseNet()
    output = net(x, y, z)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def fusion_net_compare_result():
    """
    Feature: graph kernel testcase for auto tensor inplace
    Description: random input when using graph_kernel in graph mode
    Expectation: get the same result when using and not using auto tensor inplace
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.random.normal(0, 1, [4, 16]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [16, 4]).astype(np.float32))
    z = Tensor(np.random.normal(0, 1, [4, 4]).astype(np.float32))
    expect = fusion_net_get_output(x, y, z, False)
    output = fusion_net_get_output(x, y, z, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 1.e-4, 1.e-7)
