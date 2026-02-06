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
"""Test sequential cell."""
import numpy as np
import torch
from mindspore import Tensor
from mindspore import nn


def test_sequentialcell_input_list_conv2d_bn_relu():
    """
    Feature: test sequential cell.
    Description: test sequential cell.
    Expectation: the result match with expected result.
    """
    input_np = np.random.random((1, 3, 4, 4)).astype(np.float32)
    weight_np = np.ones((2, 3, 3, 3)).astype(np.float32) * 0.000001
    bias_np = np.ones(2).astype(np.float32) * 0.00001

    input_me = Tensor(input_np)
    weight = torch.from_numpy(weight_np.astype(np.float32))
    bias = torch.from_numpy(bias_np.astype(np.float32))

    conv = nn.Conv2d(
        3, 2, 3, has_bias=False, weight_init=Tensor(weight_np), pad_mode="valid"
    )
    bn = nn.BatchNorm2d(2)
    relu = nn.ReLU()
    seq = nn.SequentialCell([conv, bn, relu])
    out_me = seq(input_me)

    net = torch.nn.Conv2d(3, 2, 3)
    net.register_parameter("weight", torch.nn.Parameter(weight))
    net.register_parameter("bias", torch.nn.Parameter(bias))
    seq_torch = torch.nn.Sequential(net, torch.nn.BatchNorm2d(2), torch.nn.ReLU())
    input_pt = torch.from_numpy(input_np.copy().astype(np.float32))
    out_torch = seq_torch(input_pt)

    np.allclose(out_me.asnumpy(), out_torch.detach().numpy(), 0.005, 0.005)
