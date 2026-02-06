# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test cell state dict."""
import numpy as np
import torch
import mindspore as ms
from mindspore import nn, Tensor, Parameter


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer1", Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer1", Tensor(np.array([4, 5, 6]).astype(np.float32)))
        self.register_buffer("buffer2", Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer3", Tensor(np.array([1, 2, 3, 4]).astype(np.float32)), persistent=True)
        self.register_buffer("buffer4", Tensor(np.array([1, 2, 3, 4, 5]).astype(np.float32)), persistent=False)
        self.register_buffer("buffer5", None, persistent=False)
        self.register_buffer("buffer6", None, persistent=True)
        self.tmp = Tensor(np.array([2, 2, 3]).astype(np.float32))
        self.register_buffer("buffer7", self.tmp)
        self.register_buffer("buffer8", self.tmp)
        self.weight1 = Parameter(Tensor(np.ones((1, 2)), ms.float32), name="w1", requires_grad=True)
        self.weight2 = Parameter(Tensor(np.ones((3, 4)), ms.float32), name="w2", requires_grad=False)

    def construct(self, x):
        return x + self.buffer2


class NetTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer1", torch.Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer1", torch.Tensor(np.array([4, 5, 6]).astype(np.float32)))
        self.register_buffer("buffer2", torch.Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.register_buffer("buffer3", torch.Tensor(np.array([1, 2, 3, 4]).astype(np.float32)), persistent=True)
        self.register_buffer("buffer4", torch.Tensor(np.array([1, 2, 3, 4, 5]).astype(np.float32)), persistent=False)
        self.register_buffer("buffer5", None, persistent=False)
        self.register_buffer("buffer6", None, persistent=True)
        self.tmp = torch.Tensor(np.array([2, 2, 3]).astype(np.float32))
        self.register_buffer("buffer7", self.tmp)
        self.register_buffer("buffer8", self.tmp)
        self.weight1 = torch.nn.Parameter(torch.Tensor(np.ones((1, 2)).astype(np.float32)), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.Tensor(np.ones((3, 4)).astype(np.float32)), requires_grad=False)

    def forward(self, x):
        return x + self.buffer2


def test_state_dict_with_one_cell():
    """
    Feature: test state dict.
    Description: test state dict.
    Expectation: the result match with expected result.
    """
    net = Net()
    net_torch = NetTorch()
    torch_state_dict = net_torch.state_dict()
    ms_state_dict = net.state_dict()
    for k, v in ms_state_dict.items():
        assert np.array_equal(v, torch_state_dict[k]), "state_dict not equal, please check"
    assert len(torch_state_dict) == len(ms_state_dict)

    net.register_buffer("buffer9", Tensor(np.array([10, 2, 3, 4]).astype(np.float32)), persistent=True)
    net.register_buffer("buffer10", Tensor(np.array([10, 2, 3, 4, 5]).astype(np.float32)), persistent=False)
    net_torch.register_buffer("buffer9", torch.Tensor(np.array([10, 2, 3, 4]).astype(np.float32)), persistent=True)
    net_torch.register_buffer("buffer10", torch.Tensor(np.array([10, 2, 3, 4, 5]).astype(np.float32)), persistent=False)
    torch_state_dict = net_torch.state_dict()
    ms_state_dict = net.state_dict()
    for k, v in ms_state_dict.items():
        assert np.array_equal(v, torch_state_dict[k]), "state_dict not equal, please check"
    assert len(torch_state_dict) == len(ms_state_dict)

    torch_dict = {}
    net_torch.state_dict(destination=torch_dict)
    ms_dict = {}
    net.state_dict(destination=ms_dict)
    for k, v in ms_dict.items():
        assert np.array_equal(v, torch_dict[k]), "state_dict not equal, please check"
    assert len(torch_dict) == len(ms_dict)

    torch_state_dict = net_torch.state_dict(prefix="test")
    ms_state_dict = net.state_dict(prefix="test")
    for k, v in ms_state_dict.items():
        assert np.array_equal(v, torch_state_dict[k]), "state_dict not equal, please check"
    assert len(torch_state_dict) == len(ms_state_dict)

    ms_state_dict_keep_vars = net.state_dict(destination=ms_dict, prefix="test", keep_vars=True)
    for k, v in ms_state_dict.items():
        assert np.array_equal(v, ms_state_dict_keep_vars[k]), "state_dict not equal, please check"
