# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap


class Net(nn.Cell):
    def __init__(self, _shape):
        super(Net, self).__init__()
        self.shape = _shape
        self.scatternd = P.ScatterNd()

    def construct(self, indices, update):
        return self.scatternd(indices, update, self.shape)


def scatternd_net(indices, update, _shape, expect):
    scatternd = Net(_shape)
    output = scatternd(Tensor(indices), Tensor(update))
    error = np.ones(shape=output.asnumpy().shape) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


def scatternd_positive(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    arr_indices = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int16)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)

    arr_indices = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)

    arr_indices = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int64)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)


def scatternd_negative(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    arr_indices = np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]]).astype(np.int16)
    arr_update = np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 0.],
                       [-21.4, -3.1]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)

    arr_indices = np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]]).astype(np.int32)
    arr_update = np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 0.],
                       [-21.4, -3.1]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)

    arr_indices = np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]]).astype(np.int64)
    arr_update = np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 0.],
                       [-21.4, -3.1]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)


def scatternd_positive_uint(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    arr_indices = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 5.3, 3.8, 1.2]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 12.],
                       [0., 1.]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)

    arr_indices = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int64)
    arr_update = np.array([3.2, 1.1, 5.3, 3.8, 1.2]).astype(nptype)
    shape = (2, 2)
    expect = np.array([[0., 12.],
                       [0., 1.]]).astype(nptype)
    scatternd_net(arr_indices, arr_update, shape, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_float64():
    """
    Feature: ScatterNd
    Description: statternd with float64 dtype
    Expectation: success
    """
    scatternd_positive(np.float64)
    scatternd_negative(np.float64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_float32():
    """
    Feature: ScatterNd
    Description: statternd with flaot32 dtype
    Expectation: success
    """
    scatternd_positive(np.float32)
    scatternd_negative(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_float16():
    """
    Feature: ScatterNd
    Description: statternd with flaot32 dtype
    Expectation: success
    """
    scatternd_positive(np.float16)
    scatternd_negative(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_int64():
    """
    Feature: ScatterNd
    Description: statternd with int64 dtype
    Expectation: success
    """
    scatternd_positive(np.int64)
    scatternd_negative(np.int64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_int32():
    """
    Feature: ScatterNd
    Description: statternd with int16 dtype
    Expectation: success
    """
    scatternd_positive(np.int32)
    scatternd_negative(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_int16():
    """
    Feature: ScatterNd
    Description: statternd with int16 dtype
    Expectation: success
    """
    scatternd_positive(np.int16)
    scatternd_negative(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_int8():
    """
    Feature: ScatterNd
    Description: statternd with int16 dtype
    Expectation: success
    """
    scatternd_positive(np.int8)
    scatternd_negative(np.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_uint64():
    """
    Feature: ScatterNd
    Description: statternd positive value of uint64 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_uint32():
    """
    Feature: ScatterNd
    Description: statternd positive value of uint32 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_uint16():
    """
    Feature: ScatterNd
    Description: statternd positive value of uint16 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_uint8():
    """
    Feature: ScatterNd
    Description: statternd positive value of uint8 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint8)


def vmap_1_batch():
    def calc(indices, updates, shape):
        return Net(shape)(indices, updates)

    def vmap_calc(indices, updates, shape):
        return vmap(calc, in_axes=(0, 0, None))(indices, updates, shape)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    indices1 = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32)
    update1 = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    expect1 = np.array([[0., 5.3],
                        [0., 1.1]]).astype(np.float32)
    indices2 = np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]]).astype(np.int32)
    update2 = np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(np.float32)
    expect2 = np.array([[0., 0.],
                        [-21.4, -3.1]]).astype(np.float32)
    indices = np.stack([indices1, indices2])
    updates = np.stack([update1, update2])
    shape = (2, 2)
    expect = np.stack([expect1, expect2])
    output = vmap_calc(Tensor(indices), Tensor(updates), shape).asnumpy()

    error = np.ones(shape=output.shape) * 1.0e-6
    diff = output - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_vmap():
    """
    Feature: ScatterNd
    Description: statternd vmap with 1 batch dim
    Expectation: success
    """
    vmap_1_batch()


class FunctionalNet(nn.Cell):
    def __init__(self, _shape):
        super(FunctionalNet, self).__init__()
        self.shape = _shape

    def construct(self, indices, update):
        return ops.scatter_nd(indices, update, self.shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_functional():
    """
    Feature: ScatterNd
    Description: statternd functional interface, graph mode
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    indices = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32)
    update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    shape = (2, 2)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(np.float32)
    scatternd = FunctionalNet(shape)
    output = scatternd(Tensor(indices), Tensor(update)).asnumpy()
    error = np.ones(shape=output.shape) * 1.0e-6
    diff = output - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_functional_pynative():
    """
    Feature: ScatterNd
    Description: statternd functional interface, pynative mode
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    indices = Tensor(np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32))
    update = Tensor(np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32))
    shape = (2, 2)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(np.float32)

    output = ops.scatter_nd(indices, update, shape).asnumpy()
    error = np.ones(shape=output.shape) * 1.0e-6
    diff = output - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_scatternd_cpu_onnx():
    """
    Feature: test ScatterNd op in cpu.
    Description: test the ops export onnx.
    Expectation: expect correct shape result.
    """
    import os
    import stat
    import onnxruntime
    from mindspore.train.serialization import export

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    shape = (4, 4, 4)
    net = Net(shape)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[[1, 1, 1, 1], [2, 2, 2, 2],
                         [3, 3, 3, 3], [4, 4, 4, 4]],
                        [[1, 1, 1, 1], [2, 2, 2, 2],
                         [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
    out_ms = net(Tensor(indices), Tensor(updates)).asnumpy()
    file = 'scatternd.onnx'
    export(net, Tensor(indices), Tensor(updates), file_name=file, file_format="ONNX")
    assert os.path.exists(file)

    sess = onnxruntime.InferenceSession(file)
    input_indices = sess.get_inputs()[0].name
    input_updates = sess.get_inputs()[1].name
    result = sess.run([], {input_indices: indices, input_updates: updates})[0]
    assert np.all(out_ms == result)

    os.chmod(file, stat.S_IWRITE)
    os.remove(file)
