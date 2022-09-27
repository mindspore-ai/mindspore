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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import dtype
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetCeil(nn.Cell):
    def __init__(self):
        super(NetCeil, self).__init__()
        self.ceil = P.Ceil()

    def construct(self, x):
        return self.ceil(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ceil_fp32():
    """
    Feature: Ceil gpu kernel
    Description: test the ceil.
    Expectation: match to np benchmark.
    """
    ceil = NetCeil()
    x = np.random.rand(3, 8).astype(np.float32)
    output = ceil(Tensor(x, dtype=dtype.float32))
    expect = np.ceil(x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ceil_fp16():
    """
    Feature: Ceil gpu kernel
    Description: test the ceil.
    Expectation: match to np benchmark.
    """
    ceil = NetCeil()
    x = np.random.rand(3, 8).astype(np.float16)
    output = ceil(Tensor(x, dtype=dtype.float16))
    expect = np.ceil(x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_ceil():
    """
    Feature: ALL TO ALL
    Description:  test cases for ceil in pynative mode cpu backend.
    Expectation: the result match numpy ceil
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([1.1, -2.1]).astype(np.float32))
    np_x = np.array([1.1, -2.1]).astype(np.float32)
    output = x.ceil()
    expect = np.ceil(np_x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_func_ceil():
    """
    Feature: ALL TO ALL
    Description:  test cases for ceil in pynative mode cpu backend.
    Expectation: the result match numpy ceil
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([1.1, -2.1]).astype(np.float32))
    np_x = np.array([1.1, -2.1]).astype(np.float32)
    output = F.ceil(x)
    expect = np.ceil(np_x)
    assert np.allclose(output.asnumpy(), expect)



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap():
    """
    Feature: ceil vmap.
    Description: test the rightness of ceil vmap feature.
    Expectation: Success.
    """

    def cal_ceil(x):
        return P.Ceil()(x)

    np_x = np.array([[[1.1, 0.9], [2.2, 1.8]], [[4.6, 1.3], [2.4, 2.6]],
                     [[1.0, 1.0], [2.0, 2.7]], [[1.3, 1.7], [2.9, 2.8]],
                     [[1.1, 1.4], [2.6, 2.0]], [[1.2, 1.4], [2.0, 2.4]],
                     [[1.5, 1.4], [2.3, 2.0]], [[1.8, 1.0], [2.9, 2.0]]]).astype(np.float32)
    x = Tensor(np_x)
    expect = np.ceil(np_x)

    vmap_ceil = vmap(cal_ceil, in_axes=(0), out_axes=0)
    output = vmap_ceil(x)
    assert np.allclose(output.asnumpy(), expect)



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap2():
    """
    Feature: ceil vmap.
    Description: test the rightness of ceil vmap feature.
    Expectation: Success.
    """
    def cal_ceil(x):
        return P.Ceil()(x)

    np_x = np.array([[[1.1, 0.9], [2.2, 1.8]], [[4.6, 1.3], [2.4, 2.6]],
                     [[1.0, 1.0], [2.0, 2.7]], [[1.3, 1.7], [2.9, 2.8]],
                     [[1.1, 1.4], [2.6, 2.0]], [[1.2, 1.4], [2.0, 2.4]],
                     [[1.5, 1.4], [2.3, 2.0]], [[1.8, 1.0], [2.9, 2.0]]]).astype(np.float32)
    x = Tensor(np_x)
    expect = np.ceil(np_x)
    vmap_ceil = vmap(vmap(cal_ceil, in_axes=(0), out_axes=0), in_axes=(0), out_axes=0)
    output = vmap_ceil(x)
    assert np.allclose(output.asnumpy(), expect)
