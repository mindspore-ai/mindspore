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

import pytest
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations._grad_ops as Grad


class PadV3GradNet(nn.Cell):
    def __init__(self, mode):
        super(PadV3GradNet, self).__init__()
        self.op = Grad.PadV3Grad(mode)

    def construct(self, x, paddings):
        return self.op(x, paddings)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3grad_circular_3d():
    """
    Feature: test PadV3Grad
    Description: test PadV3Grad circular mode.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    net = PadV3GradNet('circular')

    x = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mindspore.float32).reshape((1, 2, 4))
    paddings = Tensor(np.array([1, 1], dtype=np.int32))

    output = net(x, paddings)
    expect = np.array([[[6, 4], [14, 12]]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, output.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3grad_circular_4d():
    """
    Feature: test PadV3Grad
    Description: test PadV3Grad circular mode.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    net = PadV3GradNet('circular')

    x = Tensor(np.arange(18).reshape(1, 1, 3, 6).astype(np.float32))
    paddings = Tensor(np.array([1, 2, 1, 0], dtype=np.int64))

    output = net(x, paddings)
    expect = np.array([[[[17., 19., 15.], [34., 38., 30.]]]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, output.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3grad_circular_5d():
    """
    Feature: test PadV3Grad
    Description: test PadV3Grad circular mode.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    net = PadV3GradNet('circular')

    x = Tensor(np.arange(80).reshape(1, 1, 5, 4, 4).astype(np.float64))
    paddings = Tensor(np.array([1, 0, 1, 1, 2, 1], dtype=np.int64))

    output = net(x, paddings)
    expect = np.array([[[[[246., 252., 498.], [222., 228., 450.]],
                         [[164., 168., 332.], [148., 152., 300.]]]]]).astype(np.float64)
    np.testing.assert_almost_equal(expect, output.asnumpy())
