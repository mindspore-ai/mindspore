# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore import context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.trunc = P.Trunc()

    def construct(self, x0):
        return self.trunc(x0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test16_net():
    x = Tensor(np.array([1.2, -2.6, 5.0, 2.8, 0.2, -1.0, 2, -1.3, -0.4]), mstype.float16)
    uniq = Net()
    output = uniq(x)
    expect_x_result = [1., -2., 5., 2., 0., -1., 2, -1, -0]

    assert (output.asnumpy() == expect_x_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test32_net():
    x = Tensor(np.array([1.2, -2.6, 5.0, 2.8, 0.2, -1.0, 2, -1.3, -0.4]), mstype.float32)
    uniq = Net()
    output = uniq(x)
    expect_x_result = [1., -2., 5., 2., 0., -1., 2, -1, -0]

    assert (output.asnumpy() == expect_x_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def testint8_net():
    x = Tensor(np.array([1, -2, 5, 2, 0, -1, 2, -1, -0]), mstype.int8)
    uniq = Net()
    output = uniq(x)
    expect_x_result = [1, -2, 5, 2, 0, -1, 2, -1, -0]

    assert (output.asnumpy() == expect_x_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def testuint8_net():
    x = Tensor(np.array([1, 5, 2, 0]), mstype.uint8)
    uniq = Net()
    output = uniq(x)
    expect_x_result = [1, 5, 2, 0]

    assert (output.asnumpy() == expect_x_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def testint32_net():
    x = Tensor(np.array([1, -2, 5, 2, 0, -1, 2, -1, -0]), mstype.int32)
    uniq = Net()
    output = uniq(x)
    expect_x_result = [1, -2, 5, 2, 0, -1, 2, -1, -0]

    assert (output.asnumpy() == expect_x_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_trunc():
    """
    Feature: Trunc cpu op vmap feature.
    Description: test the vmap feature of Trunc.
    Expectation: success.
    """
    def manually_batched(func, inp):
        out_manual = []
        for i in range(inp.shape[0]):
            out = func(inp[i])
            out_manual.append(out)
        return F.stack(out_manual)

    inp = Tensor(np.arange(0, 5, 0.5).reshape(2, 5).astype(np.float32))
    net = Net()
    out_manual = manually_batched(net, inp)
    out_vmap = F.vmap(net, in_axes=0)(inp)

    assert np.array_equal(out_manual.asnumpy(), out_vmap.asnumpy())
