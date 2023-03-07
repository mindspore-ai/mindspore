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
import mindspore.ops.operations as ops
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore import context

np.random.seed(3)


class MSDynRankNet(Cell):
    def __init__(self, is_training=True):
        super(MSDynRankNet, self).__init__()
        self.is_training = is_training
        self.batch_norm = ops.BatchNorm(is_training=self.is_training)
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.relu = ops.ReLU()

    def construct(self, input_x, scale, offset, mean, variance, indices):
        unique_indices = self.relu(indices)
        reduced_in = self.reduce_mean(input_x, unique_indices)
        reduced_scale = self.reduce_mean(scale, unique_indices)
        reduced_offset = self.reduce_mean(offset, unique_indices)
        reduced_mean = self.reduce_mean(mean, unique_indices)
        reduced_variance = self.reduce_mean(variance, unique_indices)
        out, _, _, _, _ = self.batch_norm(reduced_in, reduced_scale, reduced_offset, reduced_mean, reduced_variance)
        return out


class NetFactory:
    def __init__(self, x, scale, offset, mean, variance, indices, dtype=np.float32, is_training=False):
        super(NetFactory, self).__init__()
        self.x = x
        self.scale = scale
        self.offset = offset
        self.mean = mean
        self.variance = variance
        self.indices = indices
        self.dtype = dtype
        self.is_training = is_training
        self.nh2nc = [0, 3, 1, 2]
        self.nc2nh = [0, 2, 3, 1]

    def mindspore_case(self):
        ms_x = Tensor(self.x)
        ms_indices = Tensor(self.indices)
        ms_scale = Tensor(self.scale)
        ms_offset = Tensor(self.offset)
        ms_mean = Tensor(self.mean)
        ms_variance = Tensor(self.variance)

        ms_dyn_x = Tensor(shape=[None for _ in ms_x.shape], dtype=ms_x.dtype)
        ms_dyn_scale = Tensor(shape=[None for _ in ms_scale.shape], dtype=ms_scale.dtype)
        ms_dyn_offset = Tensor(shape=[None for _ in ms_offset.shape], dtype=ms_offset.dtype)
        ms_dyn_mean = Tensor(shape=[None for _ in ms_mean.shape], dtype=ms_mean.dtype)
        ms_dyn_variance = Tensor(shape=[None for _ in ms_variance.shape], dtype=ms_variance.dtype)
        ms_dyn_indices = Tensor(shape=[None], dtype=ms_indices.dtype)

        ms_net = MSDynRankNet(is_training=self.is_training)
        ms_net.set_inputs(ms_dyn_x, ms_dyn_scale, ms_dyn_offset, ms_dyn_mean, ms_dyn_variance, ms_dyn_indices)

        ms_out = ms_net(ms_x, ms_scale, ms_offset, ms_mean, ms_variance, ms_indices)
        return ms_out.asnumpy()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batch_norm_dynamic_rank():
    """
    Feature: test batch norm dynamic rank
    Description: test batch norm dynamic rank with input tensor's type float32
    Expectation: none.
    """
    input_x = np.random.randn(3, 3, 4, 3, 3).astype(np.float32)
    scale_ = np.ones((4, 4)).astype(np.float32)
    offset_ = np.ones((4, 4)).astype(np.float32)
    mean_ = np.ones((4, 4)).astype(np.float32)
    variance_ = np.ones((4, 4)).astype(np.float32)
    indices_ = np.unique(np.random.randint(1, 2, (1,)).astype(np.int32))

    # graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    graph_mode_net = NetFactory(input_x, scale=scale_, offset=offset_, mean=mean_, variance=variance_, indices=indices_,
                                dtype=np.float32)
    graph_mode_out = graph_mode_net.mindspore_case()

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    pynative_mode_net = NetFactory(input_x, scale=scale_, offset=offset_, mean=mean_, variance=variance_,
                                   indices=indices_,
                                   dtype=np.float32)
    pynative_mode_out = pynative_mode_net.mindspore_case()

    assert np.allclose(pynative_mode_out, graph_mode_out, 1e-4, 1e-4)
