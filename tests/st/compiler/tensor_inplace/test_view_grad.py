# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops.auto_generate.gen_ops_def import slice_ext_view_op as slice_ext_view
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_slice_ext_grad():
    """
    Feature: Support view grad.
    Description: Support view grad.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            input_x = ops.abs(x)
            out = slice_ext_view(input_x, 0, 0, 2, 1)
            return out

    net = Net()
    np_x = np.arange(2 * 3 *4).reshape(2, 3, 4).astype(np.float32)
    x = Tensor(np_x)

    ms.set_context(mode=ms.GRAPH_MODE)
    graph_out = net(x)
    graph_grad_out = grad(net)(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    pynative_out = net(x)
    pynative_grad_out = grad(net)(x)

    assert (graph_out == pynative_out).all()
    assert (graph_grad_out == pynative_grad_out).all()
