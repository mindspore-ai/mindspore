# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms
from  mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P


context.set_context(precompile_only=True)


class Net(nn.Cell):
    def __init__(self, _shape):
        super(Net, self).__init__()
        self.shape = _shape
        self.scatternd = P.ScatterNd()

    def construct(self, indices, update):
        return self.scatternd(indices, update, self.shape)


def scatternd_net(indices, update, _shape, expect):
    scatternd = Net(_shape)
    _ = scatternd(Tensor(indices), Tensor(update))
    #error = np.ones(shape=output.asnumpy().shape) * 1.0e-6
    #diff = output.asnumpy() - expect
    #assert np.all(diff < error)
    #assert np.all(-diff < error)


def scatternd_positive(nptype, mode):
    context.set_context(mode=mode)

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


def scatternd_negative(nptype, mode):
    context.set_context(mode=mode)

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


def scatternd_positive_uint(nptype, mode):
    context.set_context(mode=mode)

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
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_float64(mode):
    """
    Feature: ScatterNd
    Description: statternd with float64 dtype
    Expectation: success
    """
    scatternd_positive(np.float64, mode)
    scatternd_negative(np.float64, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_float32(mode):
    """
    Feature: ScatterNd
    Description: statternd with float32 dtype
    Expectation: success
    """
    scatternd_positive(np.float32, mode)
    scatternd_negative(np.float32, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_float16(mode):
    """
    Feature: ScatterNd
    Description: statternd with float32 dtype
    Expectation: success
    """
    scatternd_positive(np.float16, mode)
    scatternd_negative(np.float16, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_int64(mode):
    """
    Feature: ScatterNd
    Description: statternd with int64 dtype
    Expectation: success
    """
    scatternd_positive(np.int64, mode)
    scatternd_negative(np.int64, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_int32(mode):
    """
    Feature: ScatterNd
    Description: statternd with int16 dtype
    Expectation: success
    """
    scatternd_positive(np.int32, mode)
    scatternd_negative(np.int32, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_int16(mode):
    """
    Feature: ScatterNd
    Description: statternd with int16 dtype
    Expectation: success
    """
    scatternd_positive(np.int16, mode)
    scatternd_negative(np.int16, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_int8(mode):
    """
    Feature: ScatterNd
    Description: statternd with int16 dtype
    Expectation: success
    """
    scatternd_positive(np.int8, mode)
    scatternd_negative(np.int8, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_uint64(mode):
    """
    Feature: ScatterNd
    Description: statternd positive value of uint64 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint64, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_uint32(mode):
    """
    Feature: ScatterNd
    Description: statternd positive value of uint32 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint32, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_uint16(mode):
    """
    Feature: ScatterNd
    Description: statternd positive value of uint16 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint16, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_scatternd_uint8(mode):
    """
    Feature: ScatterNd
    Description: statternd positive value of uint8 dtype
    Expectation: success
    """
    scatternd_positive_uint(np.uint8, mode)
