# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops.operations import _inner_ops as inner
import mindspore.nn as nn
import mindspore.context as context

def sequence_mask(x, maxlen):
    return C.sequence_mask(Tensor(x.astype(np.int32)), maxlen)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sequence_mask_1d():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a = np.array([2, 3, 1])
    maxlen = 4
    ms_out = sequence_mask(a, maxlen)
    expected_out = Tensor(np.array([[True, True, False, False],
                                    [True, True, True, False],
                                    [True, False, False, False]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sequence_mask_2d():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a = np.array([[0, 1, 3, 2], [1, 4, 4, 2]])
    maxlen = 6
    ms_out = sequence_mask(a, maxlen)
    expected_out = Tensor(np.array([[[False, False, False, False, False, False],
                                     [True, False, False, False, False, False],
                                     [True, True, True, False, False, False],
                                     [True, True, False, False, False, False]],
                                    [[True, False, False, False, False, False],
                                     [True, True, True, True, False, False],
                                     [True, True, True, True, False, False],
                                     [True, True, False, False, False, False]]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sequence_mask_3d():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a = np.array([[[2, 2], [1, 1]],
                  [[2, 0], [2, 1]],
                  [[0, 0], [0, 0]]])
    maxlen = 2
    ms_out = sequence_mask(a, maxlen)
    expected_out = Tensor(np.array([[[[True, True], [True, True]], [[True, False], [True, False]]],
                                    [[[True, True], [False, False]], [[True, True], [True, False]]],
                                    [[[False, False], [False, False]], [[False, False], [False, False]]]]))

    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sequence_mask_maxlen_1():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a = np.array([[[0, 1], [1, 1]],
                  [[1, 0], [1, 1]],
                  [[0, 1], [0, 1]]])
    maxlen = 1
    ms_out = sequence_mask(a, maxlen)
    expected_out = Tensor(np.array([[[[False], [True]], [[True], [True,]]],
                                    [[[True], [False]], [[True], [True]]],
                                    [[[False], [True]], [[False], [True]]]]))

    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sequence_mask_dynamic():
    class SequenceMaskDynamicNet1(nn.Cell):
        def __init__(self, maxlen):
            super(SequenceMaskDynamicNet1, self).__init__()
            self.maxlen = maxlen
            self.convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()

        def construct(self, x):
            converted_to_dynamic_shape = self.convert_to_dynamic_shape(x)
            return C.sequence_mask(converted_to_dynamic_shape, self.maxlen)

    class SequenceMaskDynamicNet2(nn.Cell):
        def __init__(self):
            super(SequenceMaskDynamicNet2, self).__init__()
            self.convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()

        def construct(self, x):
            converted_to_dynamic_shape = self.convert_to_dynamic_shape(x)
            return C.sequence_mask(converted_to_dynamic_shape)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    sequence_mask_net = SequenceMaskDynamicNet1(4)

    a = Tensor(np.array([0, 1, 0, 2, 0, 5]))
    ms_out = sequence_mask_net(a)
    expected_out = Tensor(np.array([[False, False, False, False],
                                    [True, False, False, False],
                                    [False, False, False, False],
                                    [True, True, False, False],
                                    [False, False, False, False],
                                    [True, True, True, True]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

    a = Tensor(np.array([[4, 3, 0], [0, 1, 3]]))
    ms_out = sequence_mask_net(a)
    expected_out = Tensor(np.array([[[True, True, True, True],
                                     [True, True, True, False],
                                     [False, False, False, False]],
                                    [[False, False, False, False],
                                     [True, False, False, False],
                                     [True, True, True, False]]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

    net_without_maxlen = SequenceMaskDynamicNet2()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.array([2, 3, 1])
    ms_out = net_without_maxlen(Tensor(a))
    expected_out = Tensor(np.array([[True, True, False],
                                    [True, True, True],
                                    [True, False, False]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())


def sequence_mask_optional(x):
    return C.sequence_mask(Tensor(x.astype(np.int32)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sequence_mask_optional_maxlen():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.array([2, 3, 1])
    ms_out = sequence_mask_optional(a)
    expected_out = Tensor(np.array([[True, True, False],
                                    [True, True, True],
                                    [True, False, False]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a = np.array([2, 3, 1])
    ms_out = sequence_mask_optional(a)
    expected_out = Tensor(np.array([[True, True, False],
                                    [True, True, True],
                                    [True, False, False]]))
    np.testing.assert_array_equal(expected_out.asnumpy(), ms_out.asnumpy())
