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
from mindspore.ops.operations import _inner_ops as inner

class NetUnique(nn.Cell):
    def __init__(self):
        super(NetUnique, self).__init__()
        self.unique = P.Unique()

    def construct(self, x):
        x_unique, x_idx = self.unique(x)
        return x_unique, x_idx


class NetUniqueDynamic(nn.Cell):
    def __init__(self):
        super(NetUniqueDynamic, self).__init__()
        self.convert = inner.GpuConvertToDynamicShape()
        self.unique = P.Unique()
        self.split = P.Split(0, 2)

    def construct(self, x):
        x_convert = self.convert(x)
        x_unique, x_idx = self.unique(x_convert)
        x_split = self.split(x_unique)
        return x_unique, x_idx, x_split


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d():
    x = Tensor(np.array([4, 5, 1, 2, 3, 3, 4, 5]).astype(np.float32))
    exp_output = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    exp_idx = np.array([3, 4, 0, 1, 2, 2, 3, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_float():
    x = Tensor(np.array([0.4, 0.5, 1.23, 2.2, 12.43, 12.43, 0.4, 0.5]).astype(np.float32))
    exp_output = np.array([0.4, 0.5, 1.23, 2.2, 12.43]).astype(np.float32)
    exp_idx = np.array([0, 1, 2, 3, 4, 4, 0, 1]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_sorted():
    x = Tensor(np.array([1, 1, 2, 4, 4, 4, 7, 8, 8]).astype(np.float32))
    exp_output = np.array([1, 2, 4, 7, 8]).astype(np.float32)
    exp_idx = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_zeros():
    x = Tensor(np.zeros(1000).astype(np.float32))
    exp_output = np.zeros(1).astype(np.float32)
    exp_idx = np.zeros(1000).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_large():
    x_np1 = np.arange(100)
    x_np2 = np.arange(100, 200)
    x_np3 = np.arange(200, 300)
    x_np = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3))
    x = Tensor(x_np.astype(np.float32))
    exp_output = np.arange(300).astype(np.float32)
    exp_idx = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3)).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_half():
    x = Tensor(np.array([0.4, 0.5, 1.23, 2.2, 12.43, 12.43, 0.4, 0.5]).astype(np.float16))
    exp_output = np.array([0.4, 0.5, 1.23, 2.2, 12.43]).astype(np.float16)
    exp_idx = np.array([0, 1, 2, 3, 4, 4, 0, 1]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_sorted_half():
    x = Tensor(np.array([1, 1, 2, 4, 4, 4, 7, 8, 8]).astype(np.float16))
    exp_output = np.array([1, 2, 4, 7, 8]).astype(np.float16)
    exp_idx = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_zeros_half():
    x = Tensor(np.zeros(1000).astype(np.float16))
    exp_output = np.zeros(1).astype(np.float16)
    exp_idx = np.zeros(1000).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_large_half():
    x_np1 = np.arange(100)
    x_np2 = np.arange(100, 200)
    x_np3 = np.arange(200, 300)
    x_np = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3))
    x = Tensor(x_np.astype(np.float16))
    exp_output = np.arange(300).astype(np.float16)
    exp_idx = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3)).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_int32():
    x = Tensor(np.array([4, 5, 1, 2, 3, 3, 4, 5]).astype(np.int32))
    exp_output = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    exp_idx = np.array([3, 4, 0, 1, 2, 2, 3, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_sorted_int32():
    x = Tensor(np.array([1, 1, 2, 4, 4, 4, 7, 8, 8]).astype(np.int32))
    exp_output = np.array([1, 2, 4, 7, 8]).astype(np.int32)
    exp_idx = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_zeros_int32():
    x = Tensor(np.zeros(1000).astype(np.int32))
    exp_output = np.zeros(1).astype(np.int32)
    exp_idx = np.zeros(1000).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_large_int32():
    x_np1 = np.arange(100)
    x_np2 = np.arange(100, 200)
    x_np3 = np.arange(200, 300)
    x_np = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3))
    x = Tensor(x_np.astype(np.int32))
    exp_output = np.arange(300).astype(np.int32)
    exp_idx = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3)).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_dynamic():
    x = Tensor(np.array([4, 5, 1, 2, 3, 3, 4, 5, 6]).astype(np.float32))
    expt_unique = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    expt_index = np.array([3, 4, 0, 1, 2, 2, 3, 4, 5]).astype(np.int32)
    expt_split = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)

    x2 = Tensor(np.array([1, 1, 4, 4, 7, 8, 8]).astype(np.float32))
    expt_unique2 = np.array([1, 4, 7, 8]).astype(np.float32)
    expt_index2 = np.array([0, 0, 1, 1, 2, 3, 3]).astype(np.int32)
    expt_split2 = np.array([[1, 4], [7, 8]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUniqueDynamic()
    x_unique, x_idx, x_split = net(x)
    assert (x_unique.asnumpy() == expt_unique).all()
    assert (x_idx.asnumpy() == expt_index).all()
    for i, out in enumerate(x_split):
        assert (out.asnumpy() == expt_split[i]).all()

    x_unique2, x_idx2, x_split2 = net(x2)
    assert (x_unique2.asnumpy() == expt_unique2).all()
    assert (x_idx2.asnumpy() == expt_index2).all()
    for i, out in enumerate(x_split2):
        assert (out.asnumpy() == expt_split2[i]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_int64():
    x = Tensor(np.array([4, 5, 1, 2, 3, 3, 4, 5]).astype(np.int64))
    exp_output = np.array([1, 2, 3, 4, 5]).astype(np.int64)
    exp_idx = np.array([3, 4, 0, 1, 2, 2, 3, 4]).astype(np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    print(x_unique)
    print(x_idx)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_1d_sorted_int64():
    x = Tensor(np.array([1, 1, 2, 4, 4, 4, 7, 8, 8]).astype(np.int64))
    exp_output = np.array([1, 2, 4, 7, 8]).astype(np.int64)
    exp_idx = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4]).astype(np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_zeros_int64():
    x = Tensor(np.zeros(1000).astype(np.int64))
    exp_output = np.zeros(1).astype(np.int64)
    exp_idx = np.zeros(1000).astype(np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_large_int64():
    x_np1 = np.arange(100)
    x_np2 = np.arange(100, 200)
    x_np3 = np.arange(200, 300)
    x_np = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3))
    x = Tensor(x_np.astype(np.int64))
    exp_output = np.arange(300).astype(np.int64)
    exp_idx = np.concatenate((x_np1, x_np2, x_np3, x_np1, x_np2, x_np3, x_np1, x_np2, x_np3)).astype(np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUnique()
    x_unique, x_idx = net(x)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()
