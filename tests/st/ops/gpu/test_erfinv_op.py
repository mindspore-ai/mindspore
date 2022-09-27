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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import dtype
from scipy import special


class Erfinv(nn.Cell):
    def __init__(self):
        super(Erfinv, self).__init__()
        self.erfinv = P.Erfinv()

    def construct(self, x):
        return self.erfinv(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_erfinv():
    """
    Feature: ErfInv function.
    Description:  The Tensor of float16, float32 or float32.
    Expectation: Output tensor with the same shape as input。
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    erfinv = Erfinv()
    x = np.array([[0.5, 0.1, 0.2], [-0.5, 0.0, -0.9]]).astype(np.float16)
    y = np.array([[0.5, 0.1, 0.2], [-0.5, 0.0, -0.9]]).astype(np.float32)
    z = np.array([[0.5, 0.1, 0.2], [-0.5, 0.0, -0.9]]).astype(np.float64)
    x_out_ms = erfinv(Tensor(x, dtype=dtype.float16))
    y_out_ms = erfinv(Tensor(y, dtype=dtype.float32))
    z_out_ms = erfinv(Tensor(z, dtype=dtype.float64))
    x_out_sc = special.erfinv(x)
    y_out_sc = special.erfinv(y)
    z_out_sc = special.erfinv(z)
    assert (np.abs(x_out_ms.asnumpy() - x_out_sc) < 1e-3).all()
    assert (np.abs(y_out_ms.asnumpy() - y_out_sc) < 1e-4).all()
    assert (np.abs(z_out_ms.asnumpy() - z_out_sc) < 1e-5).all()
