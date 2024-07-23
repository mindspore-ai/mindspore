# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark


class Maximum(Cell):
    def __init__(self):
        super(Maximum, self).__init__()
        self.max = P.Maximum()

    def construct(self, inputa, inputb):
        return self.max(inputa, inputb)


class MaxmumGradNet(Cell):
    def __init__(self):
        super(MaxmumGradNet, self).__init__()
        self.maximum_grad = GradOfAllInputs(Maximum())

    def construct(self, x, y, dy):
        return self.maximum_grad(x, y, dy)


def gen_data():
    np.random.seed(0)
    input_x_np = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    input_y_np = np.random.normal(0, 1, [1]).astype(np.float32)
    input_dout_np = np.maximum(input_x_np, input_y_np).astype(np.float32)
    input_x = Tensor(input_x_np)
    input_y = Tensor(input_y_np)
    input_dout = Tensor(input_dout_np)
    return input_x, input_y, input_dout


def get_maximum_grad_output(input_x, input_y, input_dout, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = MaxmumGradNet()
    result = net(input_x, input_y, input_dout)
    return result[0].asnumpy(), result[1].asnumpy()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_maximum_grad():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x, input_y, input_dout = gen_data()
    result_off = get_maximum_grad_output(input_x, input_y, input_dout, False)
    result_on = get_maximum_grad_output(input_x, input_y, input_dout, True)
    assert np.allclose(result_on[0], result_off[0], rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(result_on[1], result_off[1], rtol=1.e-4, atol=1.e-8, equal_nan=True)
