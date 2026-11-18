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
"""Test log1p op."""
from tests.mark_utils import arg_mark

import numpy as np


from mindspore.ops.operations import _inner_ops as _private_math
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetLog1p(nn.Cell):
    def __init__(self):
        super().__init__()
        self.log1p = _private_math.Log1p()

    def construct(self, x):
        return self.log1p(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard')
def test_log1p_fp32():
    log1p = NetLog1p()
    x = np.random.rand(3, 8).astype(np.float32)
    output = log1p(Tensor(x, dtype=dtype.float32))
    expect = np.log1p(x)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect) < tol).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard')
def test_log1p_fp16():
    log1p = NetLog1p()
    x = np.random.rand(3, 8).astype(np.float16)
    output = log1p(Tensor(x, dtype=dtype.float16))
    expect = np.log1p(x)
    tol = 1e-3
    assert (np.abs(output.asnumpy() - expect) < tol).all()
