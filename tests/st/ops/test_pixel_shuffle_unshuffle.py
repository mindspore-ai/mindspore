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

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context


class PixelShuffleNet(nn.Cell):
    """PixelShuffleNet"""

    def construct(self, x):
        output = ops.pixel_shuffle(x, 2)
        return output


class PixelUnShuffleNet(nn.Cell):
    """PixelUnShuffleNet"""

    def construct(self, x):
        output = ops.pixel_unshuffle(x, 2)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_compile_max(mode):
    """
    Feature: Test PixelShuffleAndUnShuffle
    Description: Test the functionality of PixelShuffleAndUnShuffle
    Expectation: Success
    """
    context.set_context(mode=mode)
    input_x = np.arange(4 * 2 * 2).reshape((4, 2, 2))
    input_x = mindspore.Tensor(input_x, mindspore.dtype.int32)
    shufflenet = PixelShuffleNet()
    unshufflenet = PixelUnShuffleNet()
    output1 = shufflenet(input_x)
    expect_output1 = np.array([[[0, 4, 1, 5],
                                [8, 12, 9, 13],
                                [2, 6, 3, 7],
                                [10, 14, 11, 15]]])
    assert np.allclose(output1.asnumpy(), expect_output1)
    output2 = unshufflenet(output1)
    assert np.allclose(input_x.asnumpy(), output2.asnumpy())
