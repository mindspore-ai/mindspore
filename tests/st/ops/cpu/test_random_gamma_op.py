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
"""Retained random primitive coverage after legacy function removal."""
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from mindspore.ops.operations import _inner_ops
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import ops, int32


class RandomGammaTEST(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super().__init__()
        self.shape = shape
        self.random_gamma = _inner_ops.RandomGamma(seed)

    def construct(self, alpha):
        return self.random_gamma(self.shape, alpha)


class RandomGamma(nn.Cell):

    def construct(self, shape, alpha, seed=0):
        return _inner_ops.RandomGamma(seed=seed)(shape, alpha)


class RandomGammaDR(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reducesum = P.ReduceSum(keep_dims=False)

    def construct(self, shape, alpha, seed=0):
        axis = ops.randint(1, 2, (2,), dtype=int32)
        rand_axis = ops.unique(axis)[0]
        outshape = self.reducesum(shape, rand_axis)
        outalpha = self.reducesum(alpha, rand_axis)
        return _inner_ops.RandomGamma(seed=seed)(outshape, outalpha), outshape, outalpha


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_dynamic_rank():
    """
    Feature: RandomGamma cpu kernel for dynamic rank
    Description: test the random gamma alpha is a tensor.
    Expectation: match to tensorflow benchmark.
    """

    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
    shape_ = Tensor(np.random.randint(low=1, high=8, size=(2, 2)), ms.int32)
    alpha_ = Tensor(np.random.randint(low=1, high=5, size=(2, 2)), ms.float32)
    net = RandomGammaDR()
    input_dyn_shape = [None, None]
    alpha_dyn_shape = [None, None]
    net.set_inputs(Tensor(shape=input_dyn_shape, dtype=ms.int32),
                   Tensor(shape=alpha_dyn_shape, dtype=ms.float32))
    output, out_s, out_a = net(shape_, alpha_)
    expect = np.concatenate((out_s, np.shape(out_a)), axis=0)
    assert (output.shape == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("dtype", [ms.float64, ms.float32, ms.float16])
def test_random_gamma_op(dtype):
    """
    Feature: RandomGamma cpu kernel
    Description: test the random gamma alpha is a tensor.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor(np.array([3, 1, 2]), ms.int32)
    alpha = Tensor(np.array([5, 6]), ms.float32)
    gamma_test = RandomGammaTEST(shape=shape, seed=3)
    expect = np.array([3, 1, 2, 2])

    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
    output = gamma_test(alpha)
    print(output)
    assert (output.shape == expect).all()

    gamma = _inner_ops.RandomGamma(seed=3)
    output_2 = gamma(shape, alpha)
    print(output_2)
    assert (output_2.shape == expect).all()

    net = RandomGamma()
    input_dyn_shape = [None]
    alpha_dyn_shape = [None]
    net.set_inputs(Tensor(shape=input_dyn_shape, dtype=ms.int32),
                   Tensor(shape=alpha_dyn_shape, dtype=ms.float32), 1)
    output = net(shape, alpha, seed=1)
    print(output)
