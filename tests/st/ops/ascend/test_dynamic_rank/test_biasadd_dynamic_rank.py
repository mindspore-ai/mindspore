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


class MSBiasAddDynRankNet(Cell):
    def __init__(self, data_format="NCHW"):
        super(MSBiasAddDynRankNet, self).__init__()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.bias_add = P.BiasAdd(data_format=data_format)
        self.relu = P.ReLU()

    def construct(self, input_a, input_b, indices):
        relu_indices = self.relu(indices)
        reduce_a = self.reduce_sum(input_a, relu_indices)
        out = self.bias_add(reduce_a, input_b)
        return out


class TorchAddNet(torch.nn.Module):
    def __init__(self):
        super(TorchAddNet, self).__init__()
        self.keep_dims = False

    def forward(self, input_a, input_b, indices):
        relu_indices = torch.relu(indices)
        reduce_a = torch.sum(input_a, relu_indices.tolist(), keepdim=self.keep_dims)
        out = torch.add(reduce_a, input_b)
        return out


class BiasAddOpFactory:
    def __init__(self, in_shape, indices, dtype=np.float32, data_format="NCHW"):
        super(BiasAddOpFactory, self).__init__()
        self.dtype = dtype
        self.input_x = np.random.randn(*in_shape).astype(self.dtype)
        self.data_format = data_format
        self.indices = indices
        self.input_b = np.random.randn(in_shape[-1]).astype(self.dtype)
        self.loss = 1e-4

    def ms_biass_add_forward(self):
        a = Tensor(self.input_x)
        b = Tensor(self.input_b)
        indices = Tensor(self.indices)

        dyn_a = Tensor(shape=[None for _ in a.shape], dtype=a.dtype)
        dyn_b = Tensor(shape=[None for _ in b.shape], dtype=b.dtype)
        dyn_indices = Tensor(shape=[None], dtype=indices.dtype)

        ms_net = MSBiasAddDynRankNet(data_format=self.data_format)
        ms_net.set_inputs(dyn_a, dyn_b, dyn_indices)
        out = ms_net(a, b, indices)
        return out.asnumpy()

    def torch_bias_add_forward(self):
        torch_net = TorchAddNet()
        out = torch_net(torch.from_numpy(self.input_x), torch.from_numpy(self.input_b), torch.from_numpy(self.indices))
        return out.detach().numpy()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bias_add_dyn_rank():
    """
    Feature: test bias add dynamic rank
    Description: test bias add dynamic rank with input tensor's type float32
    Expectation: none.
    """
    in_shape = (16, 16, 16, 16, 16)
    indices_np = np.unique(np.random.randint(0, 2, size=3).astype(np.int32))
    factory = BiasAddOpFactory(in_shape=in_shape, indices=indices_np, dtype=np.float32, data_format="NCHW")
    ms_out = factory.ms_biass_add_forward()
    torch_out = factory.torch_bias_add_forward()

    np.allclose(ms_out, torch_out, factory.loss, factory.loss)
