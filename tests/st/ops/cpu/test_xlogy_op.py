# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import ops


class Xlogy(nn.Cell):
    def construct(self, x, y):
        out = ops.xlogy(x, y)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_xlogy_8d():
    """
    Feature: test xlogy 8d api.
    Description: test 8d inputs.
    Expectation: op can run.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x_shape = (1, 2, 1, 2, 1, 2, 1, 2)
    y_shape = (1, 2, 1, 2, 1, 2, 1, 2)
    input_x_np = np.random.rand(*x_shape).astype(np.float32)
    input_y_np = np.random.rand(*y_shape).astype(np.float32)

    input_x = Tensor(input_x_np)
    input_y = Tensor(input_y_np)
    net = Xlogy()
    output = net(input_x, input_y)
    expected = np.multiply(input_x, np.log(input_y))
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


if __name__ == '__main__':
    test_xlogy_8d()
