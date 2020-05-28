# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_conv """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor

weight = Tensor(np.ones([2, 2]))
in_channels = 3
out_channels = 64
kernel_size = 3


def test_check_conv2d_1():
    m = nn.Conv2d(3, 64, 3, bias_init='zeros')
    output = m(Tensor(np.ones([1, 3, 16, 50], dtype=np.float32)))
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


def test_check_conv2d_2():
    Tensor(np.ones([2, 2]))
    m = nn.Conv2d(3, 64, 4, has_bias=False, weight_init='normal')
    output = m(Tensor(np.ones([1, 3, 16, 50], dtype=np.float32)))
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


def test_check_conv2d_3():
    Tensor(np.ones([2, 2]))
    m = nn.Conv2d(3, 64, (3, 3))
    output = m(Tensor(np.ones([1, 3, 16, 50], dtype=np.float32)))
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


def test_check_conv2d_4():
    Tensor(np.ones([2, 2]))
    m = nn.Conv2d(3, 64, (3, 3), stride=2, pad_mode='pad', padding=4)
    output = m(Tensor(np.ones([1, 3, 16, 50], dtype=np.float32)))
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


def test_check_conv2d_bias():
    m = nn.Conv2d(3, 64, 3, bias_init='zeros')
    output = m(Tensor(np.ones([1, 3, 16, 50], dtype=np.float32)))
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))
