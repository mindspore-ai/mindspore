# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class ConcatNet(nn.Cell):
    def __init__(self, axis):
        super().__init__()
        self.concat = ops.Concat(axis)

    def construct(self, x, y):
        return self.concat((x, y))


def concat_net(input_shapes, axis, dtype, is_dyn):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    inputs = [np.random.randn(*shape).astype(dtype) for shape in input_shapes]

    net = ConcatNet(axis)
    # net = ops.Concat(axis)

    if is_dyn:
        # test dynamic shape
        x_dyn = Tensor(shape=[None] * inputs[0].ndim, dtype=mindspore.float16)
        y_dyn = Tensor(shape=[None] * inputs[1].ndim, dtype=mindspore.float16)
        net.set_inputs(x_dyn, y_dyn)

    output = net(Tensor(inputs[0]), Tensor(inputs[1]))
    expected = np.concatenate(inputs, axis)
    np.testing.assert_allclose(output.asnumpy(), expected, 0, 0)


def test_concat(is_dyn=False):
    """
    Feature: test concat operator in graph mode
    Description: test concat.
    Expectation: the result is correct
    """
    concat_net([(2, 128), (3, 128)], 0, np.float16, is_dyn)
    concat_net([(2, 128), (2, 256)], 1, np.float16, is_dyn)
    concat_net([(2, 128), (2, 256)], -1, np.float16, is_dyn)
    concat_net([(111, 222), (1111, 222)], 0, np.float16, is_dyn)
    concat_net([(3, 4, 2), (3, 4, 2)], 0, np.float16, is_dyn)
    concat_net([(3, 4, 2), (3, 4, 2)], 1, np.float16, is_dyn)
    concat_net([(3, 4, 2), (3, 4, 2)], 2, np.float16, is_dyn)
    concat_net([(3, 4, 2, 200), (3, 4, 2, 200)], 0, np.float16, is_dyn)
    concat_net([(3, 4, 2, 200), (3, 4, 2, 200)], 1, np.float16, is_dyn)
    concat_net([(3, 4, 2, 200), (3, 4, 2, 200)], 2, np.float16, is_dyn)
    concat_net([(3, 4, 2, 200), (3, 4, 2, 200)], 3, np.float16, is_dyn)
    concat_net([(2, 2, 2, 32, 2), (2, 2, 2, 32, 2)], -1, np.float16, is_dyn)
    concat_net([(2, 2, 2, 2, 32, 2), (2, 2, 2, 2, 32, 2)], -1, np.float16, is_dyn)
    print("run concat success")


def test_concat_dyn():
    """
    Feature: test dynamic concat operator in graph mode
    Description: test concat.
    Expectation: the result is correct
    """
    test_concat(True)
