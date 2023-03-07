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
import pytest
import numpy as np
import torch
from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import operations as P

np.random.seed(3)
context.set_context(mode=context.GRAPH_MODE)


class MSReduceSumNet(Cell):
    def __init__(self, keep_dims=False):
        super(MSReduceSumNet, self).__init__()
        self.reduce_sum = P.ReduceSum(keep_dims=keep_dims)
        self.reduce = P.ReduceSum(keep_dims=False)

    def construct(self, x, indices, axis):
        x = self.reduce(x, axis)
        return self.reduce_sum(x, indices)


class TorchReduceSumNet(torch.nn.Module):
    def __init__(self, keep_dims=False):
        super(TorchReduceSumNet, self).__init__()
        self.keep_dims = keep_dims

    def forward(self, input_x, indices, axis):
        x = torch.sum(input_x, axis.tolist(), False)
        out = torch.sum(x, indices.tolist(), self.keep_dims)
        return out


class ReduceOpFactory:
    def __init__(self, input_x, indices, axis, keep_dims, dtype=np.float32, loos=1e-4):
        super(ReduceOpFactory, self).__init__()
        self.out_grad = None
        self.input_x = input_x
        self.indices = indices
        self.axis = axis
        self.keep_dims = keep_dims
        self.dtype = dtype
        self.loss = loos

    def ms_reduce_sum_forward(self):
        net = MSReduceSumNet(self.keep_dims)
        in_tensor = Tensor(self.input_x)
        in_indices = Tensor(self.indices)
        in_axis = Tensor(self.axis)

        in_tensor_dyn = Tensor(shape=[None for _ in in_tensor.shape], dtype=in_tensor.dtype)
        in_indices_dyn = Tensor(shape=[None for _ in in_indices.shape], dtype=in_indices.dtype)
        in_axis_dyn = Tensor(shape=[None for _ in in_axis.shape], dtype=in_axis.dtype)

        net.set_inputs(in_tensor_dyn, in_indices_dyn, in_axis_dyn)
        out = net(in_tensor, in_indices, in_axis)
        return out.asnumpy()

    def torch_reduce_sum_forward(self):
        net = TorchReduceSumNet(self.keep_dims)
        out = net(torch.from_numpy(self.input_x.astype(self.dtype)), self.indices, self.axis)
        return out.detach().numpy()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_reduce_sum_dyn_rank():
    """
    Feature: test reduce sum dynamic rank
    Description: test reduce sum dynamic rank with input tensor's type float32
    Expectation: none.
    """
    dtype = np.float32
    x = np.random.randn(22, 20, 28, 36, 24, 23).astype(dtype)
    indices = np.array([0, -1])
    axis = np.unique(np.random.randint(0, 2, size=5).astype(np.int32))
    factory = ReduceOpFactory(x, indices, axis, keep_dims=True, dtype=dtype, loos=1e-4)

    ms_data = factory.ms_reduce_sum_forward()
    torch_data = factory.torch_reduce_sum_forward()
    np.allclose(torch_data, ms_data, factory.loss, factory.loss)
