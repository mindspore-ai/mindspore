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

import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops.operations import random_ops
from mindspore import Tensor, context
from mindspore.common.api import _pynative_executor


class RandomShuffleNet(nn.Cell):
    def __init__(self, seed=0, seed2=0):
        super(RandomShuffleNet, self).__init__()
        self.seed = seed
        self.seed2 = seed2
        self.random_shuffle = random_ops.RandomShuffle(self.seed, self.seed2)

    def construct(self, x):
        return self.random_shuffle(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.bool, np.complex64, np.complex128,
                                   np.float64, np.float32, np.float16])
def test_random_shuffle_op_dtype(mode, dtype):
    """
    Feature: cpu RandomShuffle
    Description: test the Tensor with all supported types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    net = RandomShuffleNet(seed=1, seed2=1)
    x = Tensor(np.array([1, 2, 3, 4, 5]).astype(dtype))
    expect_shape = (5,)
    output = net(x)
    assert output.shape == expect_shape


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("shape", [(5,), (2, 3), (12, 3, 5), (3, 4, 2, 3),
                                   (3, 4, 2, 3, 4), (3, 4, 2, 3, 4, 4),
                                   (3, 4, 2, 3, 4, 5, 3)])
def test_random_shuffle_op_tensor(mode, shape):
    """
    Feature: cpu RandomShuffle
    Description: test the 0-7D Tensor.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")
    net = RandomShuffleNet(seed=3, seed2=1)
    x = Tensor(np.random.randn(*shape).astype(np.float32))
    output = net(x)
    expect_shape = shape
    assert output.shape == expect_shape


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_random_shuffle_op_scalar(mode):
    """
    Feature: cpu RandomShuffle
    Description: test the scalar Tensor.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")
    net = RandomShuffleNet(seed=3, seed2=1)
    x = Tensor(np.array(2.5).astype(np.float32))
    output = net(x)
    assert output == x


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_random_shuffle_op_dynamic_shape(mode):
    """
    Feature: cpu RandomShuffle
    Description: test the Tensor with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")
    dyn_net = RandomShuffleNet(seed=6, seed2=2)
    net = RandomShuffleNet(seed=6, seed2=2)
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    x_dyn = Tensor(shape=[None, None, None], dtype=ms.float32)
    dyn_net.set_inputs(x_dyn)
    output_dyn = dyn_net(x)
    out = net(x)
    assert (output_dyn.asnumpy() == out.asnumpy()).all()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_random_shuffle_op_exception(mode):
    """
    Feature: cpu RandomShuffle
    Description: test the Tensor with exception.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))

    with pytest.raises(TypeError):
        ops.shuffle(2, seed=3)
        _pynative_executor.sync()

    with pytest.raises(ValueError):
        ops.shuffle(x, seed=-3)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        ops.shuffle(x, seed=1.6)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        ops.shuffle(x, seed=True)
        _pynative_executor.sync()
