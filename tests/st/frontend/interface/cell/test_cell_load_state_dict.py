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
"""Test load cell state dict."""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter


class Net(nn.Cell):
    def __init__(self, net_b):
        super().__init__()
        self.net_b = net_b
        self.param_c = Parameter(Tensor(np.array([6, 6, 6]).astype(np.float32)), "netc_param1")

    def construct(self, x):
        x = self.net_b(x)
        return x + self.param_c


class NetC(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense_c = nn.Dense(3, 10, has_bias=True)
        self.param1 = Parameter(Tensor(np.array([1, 2, 3]).astype(np.float32)), "netc_param1", requires_grad=False)
        self.register_buffer("buffer_c", Tensor(np.array([4, 5, 6]).astype(np.float32)), True)

    def construct(self, x):
        return self.dense_c(x + self.param1 + self.buffer_c)


class NetB(nn.Cell):
    def __init__(self, net_c=None):
        super().__init__()
        self.net_c = net_c
        self.dense_b = nn.Dense(10, 3, has_bias=True)
        self.param1 = Parameter(Tensor(np.array([2, 2, 2]).astype(np.float32)), "netb_param1")
        self.register_buffer("buffer_b", Tensor(np.array([3, 3, 3]).astype(np.float32)))

    def construct(self, x):
        x = self.net_c(x)
        return self.dense_b(x) + self.param1 + self.buffer_b


def pre_hook(cell, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """将 state_dict 里的 'net_b.param1' 改为 'net_b.new_param1'"""
    print("Before load_state_dict, first pre hook will work")
    print(f"print state_dict as {state_dict}")


def post_hook(cell, incompatible_keys):
    print(f"Missing keys: {incompatible_keys.missing_keys}")
    print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")


def test_load_state_dict_and_hook_same():
    """
    Feature: test state dict.
    Description: test state dict.
    Expectation: the result match with expected result.
    """
    net_c = NetC()
    net_b = NetB(net_c)
    model = Net(net_b)
    ms.save_checkpoint(model.state_dict(), 'example_4.ckpt')

    new_net_c = NetC()
    new_net_b = NetB(new_net_c)
    new_model = Net(new_net_b)
    handle1 = new_model.register_load_state_dict_pre_hook(pre_hook)
    handle2 = new_model.register_load_state_dict_post_hook(post_hook)
    new_model.load_state_dict(ms.load_checkpoint('example_4.ckpt'))

    old_names = list(model.state_dict().keys())
    new_names = list(new_model.state_dict().keys())
    for i, name in enumerate(old_names):
        assert new_names[i] == name

    old_values = list(model.state_dict().values())
    new_values = list(new_model.state_dict().values())
    for i, value in enumerate(old_values):
        assert np.allclose(value.asnumpy(), new_values[i].asnumpy())
    handle1.remove()
    handle2.remove()
