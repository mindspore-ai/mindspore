# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.context as context

class RepeatElementsNet(nn.Cell):
    def __init__(self, rep, axis):
        super(RepeatElementsNet, self).__init__()
        self.rep = rep
        self.axis = axis

    def construct(self, x):
        return C.repeat_elements(x, self.rep, self.axis)


def repeat_elements(x, rep, axis):
    repeat_elements_net = RepeatElementsNet(rep, axis)
    return repeat_elements_net(Tensor(x.astype(np.int32))).asnumpy()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_1d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_1d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1)

    ms_out = repeat_elements(a, 5, 0)
    np_out = a.repeat(5, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 513, 0)
    np_out = a.repeat(513, 0)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_1d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_1d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24)

    ms_out = repeat_elements(a, 231, 0)
    np_out = a.repeat(231, 0)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_2d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_2d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1)

    ms_out = repeat_elements(a, 13, 0)
    np_out = a.repeat(13, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 13, 1)
    np_out = a.repeat(13, 1)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_2d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24).reshape(12, 2)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_2d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24).reshape(8, 3)

    ms_out = repeat_elements(a, 23, 0)
    np_out = a.repeat(23, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 23, 1)
    np_out = a.repeat(23, 1)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_3d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_3d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1)

    ms_out = repeat_elements(a, 43, 0)
    np_out = a.repeat(43, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 43, 1)
    np_out = a.repeat(43, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 43, 2)
    np_out = a.repeat(43, 2)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_3d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(60).reshape(6, 2, 5)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_3d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(60).reshape(3, 4, 5)

    ms_out = repeat_elements(a, 14, 0)
    np_out = a.repeat(14, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 14, 1)
    np_out = a.repeat(14, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 14, 2)
    np_out = a.repeat(14, 2)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_4d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_4d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1)

    ms_out = repeat_elements(a, 17, 0)
    np_out = a.repeat(17, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 17, 1)
    np_out = a.repeat(17, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 17, 2)
    np_out = a.repeat(17, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 17, 3)
    np_out = a.repeat(17, 3)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_4d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24).reshape(4, 3, 2, 1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_4d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24).reshape(2, 2, 2, 3)

    ms_out = repeat_elements(a, 23, 0)
    np_out = a.repeat(23, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 23, 1)
    np_out = a.repeat(23, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 23, 2)
    np_out = a.repeat(23, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 23, 3)
    np_out = a.repeat(23, 3)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_5d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1, 1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 4)
    np_out = a.repeat(1, 4)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_5d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1, 1)

    ms_out = repeat_elements(a, 19, 0)
    np_out = a.repeat(19, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 19, 1)
    np_out = a.repeat(19, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 19, 2)
    np_out = a.repeat(19, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 19, 3)
    np_out = a.repeat(19, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 19, 4)
    np_out = a.repeat(19, 4)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_5d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(224).reshape(8, 2, 1, 7, 2)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 4)
    np_out = a.repeat(1, 4)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_5d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(224).reshape(1, 7, 4, 4, 2)

    ms_out = repeat_elements(a, 7, 0)
    np_out = a.repeat(7, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 7, 1)
    np_out = a.repeat(7, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 7, 2)
    np_out = a.repeat(7, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 7, 3)
    np_out = a.repeat(7, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 7, 4)
    np_out = a.repeat(7, 4)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_large_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1, 1, 1, 1, 1)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 4)
    np_out = a.repeat(1, 4)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 5)
    np_out = a.repeat(1, 5)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 6)
    np_out = a.repeat(1, 6)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 7)
    np_out = a.repeat(1, 7)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_large_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1, 1, 1, 1, 1)

    ms_out = repeat_elements(a, 42, 0)
    np_out = a.repeat(42, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 1)
    np_out = a.repeat(42, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 2)
    np_out = a.repeat(42, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 3)
    np_out = a.repeat(42, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 4)
    np_out = a.repeat(42, 4)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 5)
    np_out = a.repeat(42, 5)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 6)
    np_out = a.repeat(42, 6)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 42, 7)
    np_out = a.repeat(42, 7)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_large_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1152).reshape(2, 3, 4, 8, 1, 1, 2, 3)

    ms_out = repeat_elements(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 4)
    np_out = a.repeat(1, 4)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 5)
    np_out = a.repeat(1, 5)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 6)
    np_out = a.repeat(1, 6)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 1, 7)
    np_out = a.repeat(1, 7)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_large_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1152).reshape(4, 3, 4, 2, 1, 1, 4, 3)

    ms_out = repeat_elements(a, 4, 0)
    np_out = a.repeat(4, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 1)
    np_out = a.repeat(4, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 2)
    np_out = a.repeat(4, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 3)
    np_out = a.repeat(4, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 4)
    np_out = a.repeat(4, 4)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 5)
    np_out = a.repeat(4, 5)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 6)
    np_out = a.repeat(4, 6)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 7)
    np_out = a.repeat(4, 7)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_half():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1152).astype(np.float16).reshape(4, 3, 4, 2, 1, 1, 4, 3)

    ms_out = repeat_elements(a, 4, 0)
    np_out = a.repeat(4, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 1)
    np_out = a.repeat(4, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 2)
    np_out = a.repeat(4, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 3)
    np_out = a.repeat(4, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 4)
    np_out = a.repeat(4, 4)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 5)
    np_out = a.repeat(4, 5)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 6)
    np_out = a.repeat(4, 6)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements(a, 4, 7)
    np_out = a.repeat(4, 7)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_net_multi_use():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    rep = 3
    axis = 4
    repeat_elements_net = RepeatElementsNet(rep, axis)

    a = np.arange(64).reshape(2, 2, 2, 2, 2, 2)
    ms_out = repeat_elements_net(Tensor(a.astype(np.int32))).asnumpy()
    np_out = a.repeat(rep, axis)
    np.testing.assert_array_equal(np_out, ms_out)

    a = np.arange(128).reshape(2, 2, 4, 2, 2, 2)
    ms_out = repeat_elements_net(Tensor(a.astype(np.int32))).asnumpy()
    np_out = a.repeat(rep, axis)
    np.testing.assert_array_equal(np_out, ms_out)

    a = np.arange(18).reshape(1, 1, 3, 2, 3, 1)
    ms_out = repeat_elements_net(Tensor(a.astype(np.int32))).asnumpy()
    np_out = a.repeat(rep, axis)
    np.testing.assert_array_equal(np_out, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_invalid_input():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(64).reshape(2, 2, 2, 2, 2, 2)
    with pytest.raises(ValueError):
        _ = repeat_elements(a, 0, 0)

    with pytest.raises(ValueError):
        _ = repeat_elements(a, 1, 6)

    with pytest.raises(ValueError):
        _ = repeat_elements(a, 1, -7)
