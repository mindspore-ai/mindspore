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
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore import mutable

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetMeshgrid(nn.Cell):
    def __init__(self, indexing="xy"):
        super(NetMeshgrid, self).__init__()
        self.meshgrid = P.Meshgrid(indexing)

    def construct(self, inputs):
        return self.meshgrid(inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_meshgrid_dshape():
    """
    Feature: Test meshgrid dynamic shape.
    Description: Test meshgrid dynamic shape.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetMeshgrid()
    input_x_dyn = Tensor(shape=[None], dtype=ms.float32)
    input_y_dyn = Tensor(shape=[None], dtype=ms.float32)
    net.set_inputs(mutable((input_x_dyn, input_y_dyn)))
    input_x = Tensor(np.random.random(([3])), dtype=ms.float32)
    input_y = Tensor(np.random.random(([4])), dtype=ms.float32)
    output = net(mutable((input_x, input_y)))
    expect_shape = (4, 3)
    assert output[0].asnumpy().shape == expect_shape
    assert output[1].asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype',
                         [np.bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64,
                          np.float16, np.float32, np.float64])
@pytest.mark.parametrize('indexing', ["xy", "ij"])
def test_meshgrid(dtype, indexing):
    """
    Feature: Meshgrid cpu kernel
    Description: test the rightness of Meshgrid cpu kernel
    Expectation: the output is same as np output
    """
    meshgrid = NetMeshgrid(indexing)
    x = np.random.uniform(low=0, high=10, size=3).astype(dtype)
    y = np.random.uniform(low=0, high=10, size=4).astype(dtype)
    np_output = np.meshgrid(x, y, indexing=indexing)
    output = meshgrid((Tensor(x), Tensor(y)))
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])

    # test functional interface
    output = F.meshgrid((Tensor(x), Tensor(y)), indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])

    z = np.random.uniform(low=0, high=10, size=5).astype(dtype)
    np_output = np.meshgrid(x, y, z, indexing=indexing)
    output = meshgrid((Tensor(x), Tensor(y), Tensor(z)))
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])

    # test functional interface
    output = F.meshgrid((Tensor(x), Tensor(y), Tensor(z)), indexing)
    assert np.array_equal(output[0].asnumpy(), np_output[0])
    assert np.array_equal(output[1].asnumpy(), np_output[1])
    assert np.array_equal(output[2].asnumpy(), np_output[2])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('axis', [2])
def test_meshgrid_vmap_cpu(axis):
    """
    Feature: Meshgrid cpu kernel
    Description: test the rightness of Meshgrid cpu kernel vmap feature.
    Expectation: Success.
    """
    meshgrid = NetMeshgrid("xy")

    def meshgrid_func(inputs):
        """meshgrid_func"""
        return meshgrid(inputs)

    x = np.random.uniform(low=0, high=10, size=(3, axis)).astype(np.int32)
    y = np.random.uniform(low=0, high=10, size=(4, axis)).astype(np.int32)
    inputs = (Tensor(x), Tensor(y))

    output_vmap0, output_vmap1 = vmap(meshgrid_func, in_axes=((1, 1),))(inputs)

    output_manually0 = ()
    output_manually1 = ()
    for i in range(axis):
        x_t = x[:, i]
        y_t = y[:, i]
        out0_t, out1_t = meshgrid((Tensor(x_t), Tensor(y_t)))
        output_manually0 = output_manually0 + (out0_t.asnumpy(),)
        output_manually1 = output_manually1 + (out1_t.asnumpy(),)

    assert np.array_equal(output_vmap0.asnumpy(), output_manually0)
    assert np.array_equal(output_vmap1.asnumpy(), output_manually1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('axis0', [2])
@pytest.mark.parametrize('axis1', [2])
def test_meshgrid_vmap_cpu_2(axis0, axis1):
    """
    Feature: Meshgrid cpu kernel
    Description: test the rightness of Meshgrid cpu kernel vmap feature.
    Expectation: Success.
    """
    meshgrid = NetMeshgrid("xy")

    def meshgrid_func(inputs):
        """meshgrid_func"""
        return meshgrid(inputs)

    x = np.random.uniform(low=0, high=10, size=(
        3, axis0, axis1)).astype(np.int32)
    y = np.random.uniform(low=0, high=10, size=(
        4, axis0, axis1)).astype(np.int32)
    inputs = (Tensor(x), Tensor(y))

    output_vmap0, output_vmap1 = vmap(
        vmap(meshgrid_func, in_axes=((1, 1),)), in_axes=((2, 2),))(inputs)

    output_manually0 = ()
    output_manually1 = ()
    for i in range(axis1):
        for j in range(axis0):
            x_t = x[:, j, i]
            y_t = y[:, j, i]
            out0_t, out1_t = meshgrid((Tensor(x_t), Tensor(y_t)))
            output_manually0 = output_manually0 + (out0_t.asnumpy(),)
            output_manually1 = output_manually1 + (out1_t.asnumpy(),)

    shape = (axis1, axis0) + output_manually0[0].shape
    output_manually0 = np.reshape(output_manually0, shape)
    output_manually1 = np.reshape(output_manually1, shape)

    assert np.array_equal(output_vmap0.asnumpy(), output_manually0)
    assert np.array_equal(output_vmap1.asnumpy(), output_manually1)
