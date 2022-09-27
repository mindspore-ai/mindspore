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
import pytest

import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


class InvertDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(InvertDynamicShapeNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x):
        x = self.test_dynamic(x)
        return F.invert(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.level1
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_invert(mode, shape, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for invert
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    invert = P.Invert()
    prop = 100 if np.random.random() > 0.5 else -100
    input_x = (np.random.randn(*shape) * prop).astype(dtype)
    output = invert(Tensor(input_x))
    expect_output = np.invert(input_x)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_invert_vmap(mode):
    """
    Feature: test invert vmap feature.
    Description: test invert vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[25, 4, 13, 9], [2, -1, 0, -5]], dtype=np.int16))
    # Case 1
    output = F.vmap(F.invert, 0, 0)(x)
    expect_output = np.array([[-26, -5, -14, -10], [-3, 0, -1, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(F.invert, 1, 0)(x)
    expect_output = np.array([[-26, -3], [-5, 0], [-14, -1], [-10, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    output = F.vmap(F.invert, 0, 1)(x)
    expect_output = np.array([[-26, -3], [-5, 0], [-14, -1], [-10, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_invert_dynamic_shape(mode):
    """
    Feature: test invert dynamic_shape feature.
    Description: test invert dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[25, 4, 13, 9],
                         [2, -1, 0, -5]], dtype=np.int16))
    output = InvertDynamicShapeNet()(x)
    expect_output = np.array([[-26, -5, -14, -10],
                              [-3, 0, -1, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)
