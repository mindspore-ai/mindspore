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
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
import mindspore.nn as nn
import mindspore.context as context


class RepeatElementsNet(nn.Cell):
    def __init__(self, rep, axis):
        super(RepeatElementsNet, self).__init__()
        self.repeat_elements = P.RepeatElements(rep, axis)

    def construct(self, x):
        return self.repeat_elements(x)


class RepeatElementsGradNet(nn.Cell):
    def __init__(self, rep, axis):
        super(RepeatElementsGradNet, self).__init__()
        self.repeat_elements_grad = G.RepeatElementsGrad(rep, axis)

    def construct(self, dy):
        return self.repeat_elements_grad(dy)


def repeat_elements(x, rep, axis):
    repeat_elements_net = RepeatElementsNet(rep, axis)
    return repeat_elements_net(Tensor(x.astype(np.int32))).asnumpy()


def repeat_elements_grad(dy, rep, axis):
    repeat_elements_grad_net = RepeatElementsGradNet(rep, axis)
    return repeat_elements_grad_net(Tensor(dy.astype(np.int32))).asnumpy()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_1d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1)

    ms_out = repeat_elements_grad(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_1d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1, 2)

    y = repeat_elements(a, 5, 0)
    print(y)
    ms_out = repeat_elements_grad(y, 5, 0)
    print(ms_out)
    np.testing.assert_array_equal(a*5, ms_out)

    y = repeat_elements(a, 513, 0)
    ms_out = repeat_elements_grad(y, 513, 0)
    np.testing.assert_array_equal(a*513, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_1d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24)

    ms_out = repeat_elements_grad(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_1d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(4)

    y = repeat_elements(a, 3, 0)
    ms_out = repeat_elements_grad(y, 3, 0)
    np.testing.assert_array_equal(a*3, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_2d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1)

    ms_out = repeat_elements_grad(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_2d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1)

    y = repeat_elements(a, 13, 0)
    ms_out = repeat_elements_grad(y, 13, 0)
    np.testing.assert_array_equal(a*13, ms_out)

    y = repeat_elements(a, 13, 1)
    ms_out = repeat_elements_grad(y, 13, 1)
    np.testing.assert_array_equal(a*13, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_2d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24).reshape(12, 2)

    ms_out = repeat_elements_grad(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_2d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(24).reshape(8, 3)

    y = repeat_elements(a, 23, 0)
    ms_out = repeat_elements_grad(y, 23, 0)
    np.testing.assert_array_equal(a*23, ms_out)

    y = repeat_elements(a, 23, 1)
    ms_out = repeat_elements_grad(y, 23, 1)
    np.testing.assert_array_equal(a*23, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_5d_one_element_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1, 1)

    ms_out = repeat_elements_grad(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 4)
    np_out = a.repeat(1, 4)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_5d_one_element_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1).reshape(1, 1, 1, 1, 1)

    y = repeat_elements(a, 19, 0)
    ms_out = repeat_elements_grad(y, 19, 0)
    np.testing.assert_array_equal(a, ms_out)

    y = repeat_elements(a, 19, 1)
    ms_out = repeat_elements_grad(y, 19, 1)
    np.testing.assert_array_equal(a, ms_out)

    y = repeat_elements(a, 19, 2)
    ms_out = repeat_elements_grad(y, 19, 2)
    np.testing.assert_array_equal(a, ms_out)

    y = repeat_elements(a, 19, 3)
    ms_out = repeat_elements_grad(y, 19, 3)
    np.testing.assert_array_equal(a, ms_out)

    y = repeat_elements(a, 19, 4)
    ms_out = repeat_elements_grad(y, 19, 4)
    np.testing.assert_array_equal(a, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_5d_rep_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(224).reshape(8, 2, 1, 7, 2)

    ms_out = repeat_elements_grad(a, 1, 0)
    np_out = a.repeat(1, 0)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 1)
    np_out = a.repeat(1, 1)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 2)
    np_out = a.repeat(1, 2)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 3)
    np_out = a.repeat(1, 3)
    np.testing.assert_array_equal(np_out, ms_out)

    ms_out = repeat_elements_grad(a, 1, 4)
    np_out = a.repeat(1, 4)
    np.testing.assert_array_equal(np_out, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_5d_rep_many():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(224).reshape(1, 7, 4, 4, 2)

    y = repeat_elements(a, 7, 0)
    ms_out = repeat_elements_grad(y, 7, 0)
    np.testing.assert_array_equal(a*7, ms_out)

    y = repeat_elements(a, 7, 1)
    ms_out = repeat_elements_grad(y, 7, 1)
    np.testing.assert_array_equal(a*7, ms_out)

    y = repeat_elements(a, 7, 2)
    ms_out = repeat_elements_grad(y, 7, 2)
    np.testing.assert_array_equal(a*7, ms_out)

    y = repeat_elements(a, 7, 3)
    ms_out = repeat_elements_grad(y, 7, 3)
    np.testing.assert_array_equal(a*7, ms_out)

    y = repeat_elements(a, 7, 4)
    ms_out = repeat_elements_grad(y, 7, 4)
    np.testing.assert_array_equal(a*7, ms_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_elements_grad_half():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a = np.arange(1152).astype(np.float16).reshape(4, 3, 4, 2, 1, 1, 4, 3)

    y = repeat_elements(a, 4, 0)
    ms_out = repeat_elements_grad(y, 4, 0)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 1)
    ms_out = repeat_elements_grad(y, 4, 1)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 2)
    ms_out = repeat_elements_grad(y, 4, 2)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 3)
    ms_out = repeat_elements_grad(y, 4, 3)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 4)
    ms_out = repeat_elements_grad(y, 4, 4)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 5)
    ms_out = repeat_elements_grad(y, 4, 5)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 6)
    ms_out = repeat_elements_grad(y, 4, 6)
    np.testing.assert_array_equal(a*4, ms_out)

    y = repeat_elements(a, 4, 7)
    ms_out = repeat_elements_grad(y, 4, 7)
    np.testing.assert_array_equal(a*4, ms_out)
