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
"""
test image api
"""
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import _cell_graph_executor


class PixelShuffleAndUnShuffleNet(nn.Cell):
    """PixelShuffleAndUnShuffleNet"""

    def construct(self, x):
        scaler = 3
        output_shuffle = ops.pixel_shuffle(x, scaler)
        output_unshuffle = ops.pixel_unshuffle(output_shuffle, scaler)
        return output_unshuffle


def test_compile_pixel_shuffle_unshuffle():
    """
    Feature: Test ops PixelShuffleAndUnShuffle
    Description: Test the functionality of PixelShuffleAndUnShuffle
    Expectation: Success
    """
    net = PixelShuffleAndUnShuffleNet()
    input_x = np.arange(3 * 2 * 9 * 4 * 4).reshape((3, 2, 9, 4, 4))
    input_x = mindspore.Tensor(input_x, mindspore.dtype.int32)
    _cell_graph_executor.compile(net, input_x)
