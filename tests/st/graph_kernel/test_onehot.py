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
import mindspore.common.dtype as mstype


class FusionNet(Cell):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.mul = P.Mul()

    def construct(self, x, y, indices, depth):
        res_1 = self.reshape(indices, (4096,))
        res_2 = self.onehot(res_1, depth, self.on_value, self.off_value)
        res_3 = self.mul(res_2, x)
        res_4 = self.add(res_3, y)

        return res_4


def fusion_net_get_output(x, y, indices, depth, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = FusionNet()

    output = net(x, y, indices, depth)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_graph_mode():
    """
    Feature: graph kernel testcase for onehot
    Description: random input when using graph_kernel in graph mode
    Expectation: get the same result when using and not using graph kernel
    """
    context.set_context(mode=context.GRAPH_MODE)
    depth = 512
    indices = Tensor(np.random.randint(depth, size=[4, 1024]).astype(np.int32))
    x = Tensor(np.random.normal(0, 1, [4096, 512]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [4096, 1]).astype(np.float32))
    expect = fusion_net_get_output(x, y, indices, depth, False)
    output = fusion_net_get_output(x, y, indices, depth, True)
    assert np.allclose(expect.asnumpy(), output.asnumpy(), 1.e-4, 1.e-7)
