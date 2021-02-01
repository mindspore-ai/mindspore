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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class TensorAdd(nn.Cell):
    def __init__(self):
        super(TensorAdd, self).__init__()
        self.add = P.Add()

    def construct(self, x, y):
        res = self.add(x, y)
        return res


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_add():
    x0 = Tensor(np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(np.float32))
    y0 = Tensor(np.random.uniform(-2, 2, (1, 1, 1, 1)).astype(np.float32))
    x1 = Tensor(np.random.uniform(-2, 2, (1, 3, 1, 4)).astype(np.float32))
    y1 = Tensor(np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(np.float32))
    x2 = Tensor(np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(np.float32))
    y2 = Tensor(2, mstype.float32)
    x3 = Tensor(2, mstype.float32)
    y3 = Tensor(2, mstype.float32)
    x4 = Tensor(np.random.uniform(-2, 2, (4)).astype(np.float32))
    y4 = Tensor(np.random.uniform(-2, 2, (4, 4)).astype(np.float32))
    add = TensorAdd()
    out = add(x0, y0).asnumpy()
    exp = x0.asnumpy() + y0.asnumpy()
    diff = np.abs(out - exp)
    err = np.ones(shape=exp.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == exp.shape

    out = add(x1, y1).asnumpy()
    exp = x1.asnumpy() + y1.asnumpy()
    diff = np.abs(out - exp)
    err = np.ones(shape=exp.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == exp.shape

    out = add(x2, y2).asnumpy()
    exp = x2.asnumpy() + y2.asnumpy()
    diff = np.abs(out - exp)
    err = np.ones(shape=exp.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == exp.shape

    out = add(x3, y3).asnumpy()
    exp = x3.asnumpy() + y3.asnumpy()
    diff = np.abs(out - exp)
    err = np.ones(shape=exp.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == exp.shape

    out = add(x4, y4).asnumpy()
    exp = x4.asnumpy() + y4.asnumpy()
    diff = np.abs(out - exp)
    err = np.ones(shape=exp.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == exp.shape
