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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class NetGatherV2_axis0(nn.Cell):
    def __init__(self):
        super(NetGatherV2_axis0, self).__init__()
        self.gatherv2 = P.Gather()

    def construct(self, params, indices):
        return self.gatherv2(params, indices, 0)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherv2_axis0():
    x = Tensor(np.arange(3 * 2 * 2).reshape(3, 2, 2), mstype.float32)
    indices = Tensor(np.array([1, 2]), mstype.int32)
    gatherv2 = NetGatherV2_axis0()
    ms_output = gatherv2(x, indices)
    print("output:\n", ms_output)
    expect = np.array([[[4., 5.],
                        [6., 7.]],
                       [[8., 9.],
                        [10., 11.]]])
    error = np.ones(shape=ms_output.asnumpy().shape) * 1.0e-6
    diff = ms_output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

class NetGatherV2_axis1(nn.Cell):
    def __init__(self):
        super(NetGatherV2_axis1, self).__init__()
        self.gatherv2 = P.Gather()

    def construct(self, params, indices):
        return self.gatherv2(params, indices, 1)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherv2_axis1():
    x = Tensor(np.arange(2 * 3 * 2).reshape(2, 3, 2), mstype.float32)
    indices = Tensor(np.array([1, 2]), mstype.int32)
    gatherv2 = NetGatherV2_axis1()
    ms_output = gatherv2(x, indices)
    print("output:\n", ms_output)
    expect = np.array([[[2., 3.],
                        [4., 5.]],
                       [[8., 9.],
                        [10., 11.]]])
    error = np.ones(shape=ms_output.asnumpy().shape) * 1.0e-6
    diff = ms_output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

class NetGatherV2_axisN1(nn.Cell):
    def __init__(self):
        super(NetGatherV2_axisN1, self).__init__()
        self.gatherv2 = P.Gather()

    def construct(self, params, indices):
        return self.gatherv2(params, indices, -1)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherv2_axisN1():
    x = Tensor(np.arange(2 * 2 * 3).reshape(2, 2, 3), mstype.float32)
    indices = Tensor(np.array([1, 2]), mstype.int32)
    gatherv2 = NetGatherV2_axisN1()
    ms_output = gatherv2(x, indices)
    print("output:\n", ms_output)
    expect = np.array([[[1., 2.],
                        [4., 5.]],
                       [[7., 8.],
                        [10., 11.]]])
    error = np.ones(shape=ms_output.asnumpy().shape) * 1.0e-6
    diff = ms_output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


def cal_vmap_gather(x, indices, axis):
    return P.Gather()(x, indices, axis)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_basic():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of vmap gather basic.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [1, 2]]).astype(np.int32))
    axis = 0

    outputs = vmap(cal_vmap_gather, in_axes=(0, 0, None), out_axes=0)(x, indices, axis)

    expect = np.array([[1, 2],
                       [5, 6]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_negative_axis():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of vmap gather when axis is negative.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]],
                         [[7, 8, 9],
                          [10, 11, 12]]]).astype(np.float32))
    indices = Tensor(np.array([[0, 1], [1, 0]]).astype(np.int32))
    axis = -2

    outputs = vmap(cal_vmap_gather, in_axes=(0, 0, None), out_axes=0)(x, indices, axis)

    expect = np.array([[[1, 2, 3],
                        [4, 5, 6]],
                       [[10, 11, 12],
                        [7, 8, 9]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_with_inaxes():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of vmap gather when in_axes is not zero.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]],
                         [[7, 8, 9],
                          [10, 11, 12]]]).astype(np.float32))

    x = np.moveaxis(x, 0, 2)
    indices = Tensor(np.array([[0, 1], [1, 0]]).astype(np.int32))
    indices = np.moveaxis(indices, 0, 1)
    axis = 0

    outputs = vmap(cal_vmap_gather, in_axes=(2, 1, None), out_axes=0)(x, indices, axis)

    expect = np.array([[[1, 2, 3],
                        [4, 5, 6]],
                       [[10, 11, 12],
                        [7, 8, 9]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_indices_outofbound():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of vmap gather when indices out of bound.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]],
                         [[7, 8, 9],
                          [10, 11, 12]]]).astype(np.float32))

    indices = Tensor(np.array([[0, 2], [2, 0]]).astype(np.int32))
    axis = 0

    outputs = vmap(cal_vmap_gather, in_axes=(0, 0, None), out_axes=0)(x, indices, axis)

    expect = np.array([[[1, 2, 3],
                        [0, 0, 0]],
                       [[0, 0, 0],
                        [7, 8, 9]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_xdim_is_none():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of vmap gather when no xdim.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([1, 2, 3]).astype(np.float32))

    indices = Tensor(np.array([[0, 1], [2, 0]]).astype(np.int32))
    axis = 0

    outputs = vmap(cal_vmap_gather, in_axes=(None, 0, None), out_axes=0)(x, indices, axis)

    expect = np.array([[1, 2],
                       [3, 1]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_idim_is_none():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of nested vmap gather.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))

    indices = Tensor(np.array([0, 1]).astype(np.int32))
    axis = 0

    outputs = vmap(cal_vmap_gather, in_axes=(0, None, None), out_axes=0)(x, indices, axis)

    expect = np.array([[1, 2],
                       [4, 5]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_vmap_nested():
    """
    Feature: gather vmap test on cpu.
    Description: test the rightness of nested vmap gather.
    Expectation: use vmap rule's result equal to manually batched.
    """

    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]],
                         [[7, 8, 9],
                          [10, 11, 12]]]).astype(np.float32))

    indices = Tensor(np.array([[[0, 1],
                                [1, 0]],
                               [[0, 1],
                                [1, 0]]]).astype(np.int32))
    axis = 0

    outputs = vmap(vmap(cal_vmap_gather, in_axes=(0, 0, None), out_axes=0),
                   in_axes=(0, 0, None), out_axes=0)(x, indices, axis)

    expect = np.array([[[1, 2],
                        [5, 4]],
                       [[7, 8],
                        [11, 10]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.uint64, np.uint16, np.int64, np.complex64, np.complex128])
def test_gather_tensor(data_type):
    """
    Feature: Gather
    Description: test cases for Gather on cpu
    Expectation: the result match to numpy
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7]).astype(data_type)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6], dtype=np.int))
    axis = 0
    y_expect = np.array([1, 3, 5, 3, 7]).astype(data_type)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    graph_table_tensor = Tensor(x)
    out = graph_table_tensor.gather(input_indices, axis)

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    pynative_table_tensor = Tensor(x)
    out = pynative_table_tensor.gather(input_indices, axis)

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_batch_dims():
    """
    Feature: Gather
    Description: test cases for Gather with batch_dims
    Expectation: the result match to numpy
    """
    x = np.arange(27).reshape(3, 3, 3).astype(np.int32)
    indices = np.array([[0, 0], [1, 1], [1, 1]]).astype(np.int32)
    axis = 1
    batch_dims = 1
    out = P.Gather(batch_dims)(Tensor(x), Tensor(indices), axis)
    expect = np.array([[[0, 1, 2], [0, 1, 2]],
                       [[12, 13, 14], [12, 13, 14]],
                       [[21, 22, 23], [21, 22, 23]]]).astype(np.int32)
    np.allclose(out.asnumpy(), expect)
