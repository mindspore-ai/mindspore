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
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import nn_ops


class PadV3Net(nn.Cell):
    def __init__(self, mode, paddings_contiguous=True):
        super(PadV3Net, self).__init__()
        self.ops = nn_ops.PadV3(mode, paddings_contiguous)
        self.mode = mode

    def construct(self, x, paddings, value=0):
        if self.mode == "constant":
            out = self.ops(x, paddings, value)
        else:
            out = self.ops(x, paddings)
        return out


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_padv3_circular_dynamic_shape_3d(mode):
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    context.set_context(mode=mode, device_target="Ascend", save_graphs=False)
    x = Tensor(np.arange(9).reshape(1, 3, 3).astype(np.int32))
    padding = Tensor((1, 2), dtype=ms.int64)

    net = PadV3Net('circular')

    x_dyn = Tensor(shape=(1, 3, None), dtype=x.dtype)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn)

    out = net(x, padding)
    expect = np.array([[[2, 0, 1, 2, 0, 1],
                        [5, 3, 4, 5, 3, 4],
                        [8, 6, 7, 8, 6, 7]]]).astype(np.int32)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_padv3_circular_dynamic_shape_4d(mode):
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    context.set_context(mode=mode, device_target="Ascend", save_graphs=False)
    x = Tensor(np.arange(9).reshape(1, 1, 3, 3).astype(np.float64))
    padding = Tensor((1, -1, 1, 2), dtype=ms.int32)

    net = PadV3Net('circular')

    x_dyn = Tensor(shape=(1, 1, 3, None), dtype=x.dtype)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn)

    out = net(x, padding)
    expect = np.array([[[[7, 6, 7], [1, 0, 1], [4, 3, 4],
                         [7, 6, 7], [1, 0, 1], [4, 3, 4]]]]).astype(np.float64)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_padv3_circular_dynamic_shape_5d(mode):
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    context.set_context(mode=mode, device_target="Ascend", save_graphs=False)
    x = Tensor(np.arange(18).reshape(1, 1, 2, 3, 3).astype(np.float64))
    padding = Tensor((0, 1, 1, -1, 0, -1), dtype=ms.int32)

    net = PadV3Net('circular')

    x_dyn = Tensor(shape=(1, 1, None, 3, None), dtype=x.dtype)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn)

    out = net(x, padding)
    expect = np.array([[[[[3, 4, 5, 3,],
                          [0, 1, 2, 0,],
                          [3, 4, 5, 3,]]]]]).astype(np.float64)
    np.testing.assert_almost_equal(expect, out.asnumpy())
