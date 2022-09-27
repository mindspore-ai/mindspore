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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetUniqueWithPad(nn.Cell):
    def __init__(self, pad_num):
        super(NetUniqueWithPad, self).__init__()
        self.unique_with_pad = P.UniqueWithPad()
        self.pad_num = pad_num

    def construct(self, x):
        x_unique, x_idx = self.unique_with_pad(x, self.pad_num)
        return x_unique, x_idx


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_int32():
    """
    Feature: test uniquewithpad in gpu.
    Description: test uniquewithpad forward with int32 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 5]).astype(np.int32))
    exp_output = np.array([1, 2, 3, 4, 5, 99, 99, 99]).astype(np.int32)
    exp_idx = np.array([0, 1, 1, 2, 2, 2, 3, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUniqueWithPad(99)
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_int64():
    """
    Feature: test uniquewithpad in gpu.
    Description: test uniquewithpad forward with int64 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 5]).astype(np.int64))
    exp_output = np.array([1, 2, 3, 4, 5, 99, 99, 99]).astype(np.int64)
    exp_idx = np.array([0, 1, 1, 2, 2, 2, 3, 4]).astype(np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUniqueWithPad(99)
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_float32():
    """
    Feature: test uniquewithpad in gpu.
    Description: test uniquewithpad forward with float32 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 5]).astype(np.float32))
    exp_output = np.array([1, 2, 3, 4, 5, 99, 99, 99]).astype(np.float32)
    exp_idx = np.array([0, 1, 1, 2, 2, 2, 3, 4]).astype(np.int32)
    net = NetUniqueWithPad(99.0)
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_dynamic_shape():
    """
    Feature: uniquewithpad dynamic shape test in gpu.
    Description: test the rightness of uniquewithpad dynamic shape feature.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 5, 2]).astype(np.int32))
    net = NetUniqueWithPad(0)
    input_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    net.set_inputs(input_dyn)
    output = net(x)
    expect_y_result = [1, 2, 5, 0]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_vmap():
    """
    Feature: uniquewithpad vmap test in gpu.
    Description: test the rightness of uniquewithpad vmap feature.
    Expectation: use vmap rule's result equal to manually batched.
    """

    def cal_unique_with_pad(x):
        return P.UniqueWithPad()(x, -1)

    x = Tensor(np.array([[[1, 2, 5, 2], [1, 2, 5, 2]], [[1, 2, 5, 2], [1, 2, 5, 2]]]).astype(np.int32))

    vmap_unique_with_pad = vmap(vmap(cal_unique_with_pad, in_axes=0), in_axes=0)
    outputs = vmap_unique_with_pad(x)
    expect0 = np.array([[[1, 2, 5, -1], [1, 2, 5, -1]], [[1, 2, 5, -1], [1, 2, 5, -1]]]).astype(np.int32)
    expect1 = np.array([[[0, 1, 2, 1], [0, 1, 2, 1]], [[0, 1, 2, 1], [0, 1, 2, 1]]]).astype(np.int32)
    assert np.allclose(outputs[0].asnumpy(), expect0)
    assert np.allclose(outputs[1].asnumpy(), expect1)
