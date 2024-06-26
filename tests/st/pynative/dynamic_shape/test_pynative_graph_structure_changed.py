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
import mindspore as ms
from mindspore import nn, value_and_grad
import torch
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_graph_structure_changed():
    """
    Feature: PyNative dynamic shape for Ascend.
    Description: Test PyNative dynamic shape if set kernel info.
    Expectation: The calculation result is correct and have no exceptip in process.
    """
    input_np = np.random.rand(1, 3, 2, 2).astype(np.float32)

    ms_net = nn.BatchNorm2d(num_features=3)
    ms_net.set_train()
    ms_x = ms.Tensor(input_np)
    x_dyn = ms.Tensor(shape=[None for _ in ms_x.shape], dtype=ms_x.dtype)
    ms_net.set_inputs(x_dyn)
    grad_fn = value_and_grad(ms_net)
    ms_output, _ = grad_fn(ms_x)

    torch_net = torch.nn.BatchNorm2d(num_features=3)
    torch_x = torch.tensor(input_np)
    torch_output = torch_net(torch_x)
    assert np.allclose(torch_output.detach().numpy(), ms_output.asnumpy(), 0.01, 0.01)
