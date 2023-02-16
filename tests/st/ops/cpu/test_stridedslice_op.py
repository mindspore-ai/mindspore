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

import platform
import numpy as np
import pytest


import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class StridedSlice(nn.Cell):
    def __init__(self):
        super(StridedSlice, self).__init__()
        self.stridedslice = P.StridedSlice()

    def construct(self, x):
        return self.stridedslice(x, (2, 0, 0), (3, 2, 3), (1, 1, 1))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(np.float32))
    stridedslice = StridedSlice()
    output = stridedslice(x)
    expect = [[[5., 5., 5.],
               [6., 7., 8.]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice_vmap():
    """
    Feature: Test stridedslice CPU vmap.
    Description: test vmap for stridedslice.
    Expectation: match to np benchmark.
    """

    x = Tensor(np.ones((4, 3, 5, 16)))
    stridedslice_vmap = vmap(StridedSlice(), in_axes=-1)
    output = stridedslice_vmap(x)
    expect = np.ones((16, 1, 2, 3))
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64])
def test_slice_functional_with_attr_int32(dtype):
    """
    Feature: Test strided_slice functional interface.
    Description: Test strided_slice functional interface with attr int32.
    Expectation: success.
    """
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(dtype))
    begin = Tensor(np.array([2, 0, 0]).astype(np.int32))
    end = Tensor(np.array([3, 2, 3]).astype(np.int32))
    strides = Tensor(np.array([1, 1, 1]).astype(np.int32))
    output = ops.strided_slice(x, begin, end, strides)
    expect = np.array([[[5., 5., 5.],
                        [6., 7., 8.]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64])
def test_slice_functional_with_attr_int64(dtype):
    """
    Feature: Test strided_slice functional interface.
    Description: Test strided_slice functional interface with attr int64.
    Expectation: success.
    """

    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(dtype))
    begin = Tensor(np.array([2, 0, 0]).astype(np.int64))
    end = Tensor(np.array([3, 2, 3]).astype(np.int64))
    strides = Tensor(np.array([1, 1, 1]).astype(np.int64))
    output = ops.strided_slice(x, begin, end, strides)
    expect = np.array([[[5., 5., 5.],
                        [6., 7., 8.]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_slice_functional_with_attr_int32_complex(dtype):
    """
    Feature: Test strided_slice functional interface.
    Description: Test strided_slice functional interface with attr int32.
    Expectation: success.
    """
    if platform.system() == 'Windows':
        return
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(dtype))
    begin = Tensor(np.array([2, 0, 0]).astype(np.int32))
    end = Tensor(np.array([3, 2, 3]).astype(np.int32))
    strides = Tensor(np.array([1, 1, 1]).astype(np.int32))
    output = ops.strided_slice(x, begin, end, strides)
    expect = np.array([[[5., 5., 5.],
                        [6., 7., 8.]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_slice_functional_with_attr_int64_complex(dtype):
    """
    Feature: Test strided_slice functional interface.
    Description: Test strided_slice functional interface with attr int64.
    Expectation: success.
    """
    if platform.system() == 'Windows':
        return
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(dtype))
    begin = Tensor(np.array([2, 0, 0]).astype(np.int64))
    end = Tensor(np.array([3, 2, 3]).astype(np.int64))
    strides = Tensor(np.array([1, 1, 1]).astype(np.int64))
    output = ops.strided_slice(x, begin, end, strides)
    expect = np.array([[[5., 5., 5.],
                        [6., 7., 8.]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()
