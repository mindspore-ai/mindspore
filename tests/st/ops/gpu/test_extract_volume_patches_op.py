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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.array_ops as ops


class NetExtractVolumePatches(nn.Cell):
    def __init__(self, kernel_size, strides, padding="valid"):
        super(NetExtractVolumePatches, self).__init__()
        self.extractvolumepatches = ops.ExtractVolumePatches(
            kernel_size, strides, padding)

    def construct(self, x):
        return self.extractvolumepatches(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extractvolumepatches_graph():
    """
    Feature: extractvolumepatches
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    types = [np.float16, np.float32, np.float64, np.int8, np.int16,
             np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
    for type_i in types:
        x = Tensor(np.ones([1, 1, 3, 3, 3]).astype(type_i))
        extractvolumepatches = NetExtractVolumePatches(
            [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], "VALID")
        output = extractvolumepatches(x).transpose(0, 2, 3, 4, 1)
        expect_output = np.array([[[[[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.]],
                                    [[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.]]],
                                   [[[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.]],
                                    [[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.]]]]]).astype(type_i)
        assert np.allclose(output.asnumpy(), expect_output)
        assert output.shape == expect_output.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extractvolumepatches_pynative():
    """
    Feature: extractvolumepatches
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    types = [np.float16, np.float32, np.float64, np.int8, np.int16,
             np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
    for type_i in types:
        x = Tensor(np.ones([1, 1, 3, 3, 3]).astype(type_i))
        extractvolumepatches = NetExtractVolumePatches(
            [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], "SAME")
        output = extractvolumepatches(x).transpose(0, 2, 3, 4, 1)
        expect_output = np.array([[[[[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 0., 1., 0., 1., 0., 1., 0.]],
                                    [[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 0., 1., 0., 1., 0., 1., 0.]],
                                    [[1., 1., 0., 0., 1., 1., 0., 0.],
                                     [1., 1., 0., 0., 1., 1., 0., 0.],
                                     [1., 0., 0., 0., 1., 0., 0., 0.]]],
                                   [[[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 0., 1., 0., 1., 0., 1., 0.]],
                                    [[1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1., 1.],
                                     [1., 0., 1., 0., 1., 0., 1., 0.]],
                                    [[1., 1., 0., 0., 1., 1., 0., 0.],
                                     [1., 1., 0., 0., 1., 1., 0., 0.],
                                     [1., 0., 0., 0., 1., 0., 0., 0.]]],
                                   [[[1., 1., 1., 1., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 0., 0., 0., 0.],
                                     [1., 0., 1., 0., 0., 0., 0., 0.]],
                                    [[1., 1., 1., 1., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 0., 0., 0., 0.],
                                     [1., 0., 1., 0., 0., 0., 0., 0.]],
                                    [[1., 1., 0., 0., 0., 0., 0., 0.],
                                     [1., 1., 0., 0., 0., 0., 0., 0.],
                                     [1., 0., 0., 0., 0., 0., 0., 0.]]]]]).astype(type_i)
        assert np.allclose(output.asnumpy(), expect_output)
        assert output.shape == expect_output.shape
