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

import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, axis=0, out_nums=1):
        super(Net, self).__init__()
        self.split = P.Split(axis, out_nums)

    def construct(self, x):
        return self.split(x)


class NetDynamic(nn.Cell):
    def __init__(self, axis=0, out_nums=1):
        super(NetDynamic, self).__init__()
        self.conv = inner.GpuConvertToDynamicShape()
        self.split = P.Split(axis, out_nums)

    def construct(self, x):
        x_conv = self.conv(x)
        x_split = self.split(x_conv)
        return x_split


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


def split_basic(nptype):
    x = np.array([[[1, -1, 1], [2, -2, 2]],
                  [[3, -3, 3], [4, -4, 4]],
                  [[5, -5, 5], [6, -6, 6]]]).astype(nptype)

    split_op = Net(0, 3)
    outputs = split_op(Tensor(x))
    for i, out in enumerate(outputs):
        assert (out.asnumpy() == x[i]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_float16():
    split_basic(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_float32():
    split_basic(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_float64():
    split_basic(np.float64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_int32():
    split_basic(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_uint32():
    split_basic(np.uint32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_int64():
    split_basic(np.int64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_basic_bool():
    split_basic(np.bool)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_4d():
    x_np = np.random.randn(2, 6, 4, 4).astype(np.float32)
    y = np.split(x_np, 3, axis=1)

    split_op = Net(1, 3)
    outputs = split_op(Tensor(x_np))

    for i, out in enumerate(outputs):
        assert (out.asnumpy() == y[i]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_dynamic():
    x = np.array([[[1, -1, 1], [2, -2, 2]],
                  [[3, -3, 3], [4, -4, 4]],
                  [[5, -5, 5], [6, -6, 6]]]).astype(np.float32)

    net = NetDynamic(0, 3)
    x_split = net(Tensor(x))
    for i, out in enumerate(x_split):
        assert (out.asnumpy() == x[i]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_dynamic_axis1():
    x = np.array([[[1, -1, 1], [2, -2, 2]],
                  [[3, -3, 3], [4, -4, 4]],
                  [[5, -5, 5], [6, -6, 6]]]).astype(np.int32)
    y = np.split(x, 2, axis=1)

    net = NetDynamic(1, 2)
    x_split = net(Tensor(x))
    for i, out in enumerate(x_split):
        assert (out.asnumpy() == y[i]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_dynamic_axis2():
    x = np.array([[[1, -1, 1], [2, -2, 2]],
                  [[3, -3, 3], [4, -4, 4]],
                  [[5, -5, 5], [6, -6, 6]]]).astype(np.int32)
    y = np.split(x, 3, axis=2)

    net = NetDynamic(2, 3)
    x_split = net(Tensor(x))
    for i, out in enumerate(x_split):
        assert (out.asnumpy() == y[i]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split_invalid_input():
    with pytest.raises(TypeError):
        _ = Net(0.1, 3)

    with pytest.raises(TypeError):
        _ = Net(0, 3.0)

    with pytest.raises(ValueError):
        _ = Net(0, -3)

    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    split_net = Net(2, 2)
    with pytest.raises(ValueError):
        _ = split_net(Tensor(x))

    with pytest.raises(TypeError):
        _ = split_net(x)
