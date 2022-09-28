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

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class NetExpm1(nn.Cell):
    def __init__(self):
        super(NetExpm1, self).__init__()
        self.expm1 = P.Expm1()

    def construct(self, x):
        return self.expm1(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_expm1_fp32():
    expm1 = NetExpm1()
    x = np.random.rand(3, 8).astype(np.float32)
    output = expm1(Tensor(x, dtype=dtype.float32))
    expect = np.expm1(x)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect) < tol).all()

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_expm1_fp16():
    expm1 = NetExpm1()
    x = np.random.rand(3, 8).astype(np.float16)
    output = expm1(Tensor(x, dtype=dtype.float16))
    expect = np.expm1(x)
    tol = 1e-3
    assert (np.abs(output.asnumpy() - expect) < tol).all()
