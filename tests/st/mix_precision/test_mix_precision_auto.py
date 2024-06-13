# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test network turn on mix_precision with auto mode."""

import pytest
import numpy as np
import mindspore as ms
from mindspore.amp import auto_mixed_precision
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=in_c,
                                  gamma_init='ones',
                                  beta_init='zeros',
                                  moving_mean_init='zeros',
                                  moving_var_init='ones')
        self.bn2 = nn.BatchNorm2d(num_features=out_c,
                                  gamma_init='ones',
                                  beta_init='zeros',
                                  moving_mean_init='zeros',
                                  moving_var_init='ones')
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=True,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones')
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        return x


class Net_FP16(nn.Cell):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=in_c,
                                  gamma_init='ones',
                                  beta_init='zeros',
                                  moving_mean_init='zeros',
                                  moving_var_init='ones')
        self.bn2 = nn.BatchNorm2d(num_features=out_c,
                                  gamma_init='ones',
                                  beta_init='zeros',
                                  moving_mean_init='zeros',
                                  moving_var_init='ones')
        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              has_bias=True,
                              pad_mode='same',
                              weight_init='ones',
                              bias_init='ones').to_float(ms.float16)
        self.mean = ops.ReduceMean(keep_dims=False)
        self.cast = ops.Cast()

    def construct(self, x):
        x = self.cast(x, ms.float16)
        x = self.relu(x)
        x = self.cast(x, ms.float32)
        x = self.bn1(x)
        x = self.cast(x, ms.float16)
        x = self.conv(x)
        x = self.cast(x, ms.float32)
        x = self.bn2(x)
        x = self.cast(x, ms.float16)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.cast(x, ms.float32)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_mix_precision_train_auto():
    context.set_context(mode=context.PYNATIVE_MODE)
    input_data = np.random.randn(32, 3, 224, 224).astype(np.float32)

    # auto mixed precision
    net_pynative = Net(3, 10)
    net_pynative = auto_mixed_precision(net_pynative, amp_level="auto", dtype=ms.float16)
    out_pynative = net_pynative(Tensor(input_data))

    # manual mixed precision
    net_pynative2 = Net_FP16(3, 10)
    out_pynative2 = net_pynative2(Tensor(input_data))

    assert np.allclose(out_pynative.asnumpy(), out_pynative2.asnumpy(), 0.0001, 0.0001)
