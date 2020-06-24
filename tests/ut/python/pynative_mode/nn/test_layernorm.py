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
""" test_layernorm """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype


def test_check_layer_norm_1():
    x = Tensor(np.ones([20, 5, 10, 10]), mstype.float32)
    shape1 = x.shape[1:]
    m = nn.LayerNorm(shape1, -1, 1)
    with pytest.raises(NotImplementedError):
        m(x)


def test_check_layer_norm_2():
    x = Tensor(np.ones([20, 5, 10, 10]), mstype.float32)
    shape1 = x.shape[1:]
    m = nn.LayerNorm(shape1, begin_params_axis=1)
    with pytest.raises(NotImplementedError):
        m(x)


def test_check_layer_norm_3():
    x = Tensor(np.ones([20, 5, 10, 10]), mstype.float32)
    shape1 = (10, 10)
    m = nn.LayerNorm(shape1, begin_params_axis=2)
    with pytest.raises(NotImplementedError):
        m(x)
