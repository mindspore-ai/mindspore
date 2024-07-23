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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


class Net(nn.Cell):
    def __init__(self, seed=-1):
        super(Net, self).__init__()
        self.bernoulli = F.bernoulli

    def construct(self, x, p):
        return self.bernoulli(x, p)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bernoulli():
    """
    Feature: bernoulli function
    Description: test cases for Bernoulli
    Expectation: the result matches scipy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = [32, 16, 2, 5]
    x = np.ones(x_shape).astype(np.float32)
    bernoulli = Net()
    tx = Tensor(x)
    output = bernoulli(tx, 0.5)
    # check output
    output_np = output.asnumpy()
    nonzero_count = np.count_nonzero(output_np)
    elem_count = x.size
    assert elem_count * 0.4 < nonzero_count < elem_count * 0.6


class BernoulliDynamic(nn.Cell):
    def __init__(self, seed=-1):
        super(BernoulliDynamic, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.bernoulli = F.bernoulli

    def construct(self, x, p):
        x = self.test_dynamic(x)
        p = self.test_dynamic(p)
        return self.bernoulli(x, p)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bernoulli_dynamic():
    """
    Feature: bernoulli function
    Description: test cases for Bernoulli
    Expectation: the result matches scipy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.ones([32, 16, 2, 5]).astype(np.float32)
    p = np.ones([1]).astype(np.float32) * 0.5
    net = BernoulliDynamic()

    output = net(Tensor(x), Tensor(p))
    nonzero_count = np.count_nonzero(output.asnumpy())
    elem_count = x.size
    assert elem_count * 0.4 < nonzero_count < elem_count * 0.6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bernoulli_dynamic_x_type_error():
    """
    Feature: Bernoulli function
    Description: Test cases for Bernoulli
    Expectation: Raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    with pytest.raises(TypeError):
        x = np.random.random(size=(15, 26, 2, 8, 10))
        p = Tensor(np.random.random(size=(15, 26, 2, 8, 10)))
        net = BernoulliDynamic()
        output = net(x, p=p)
        print(output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bernoulli_run_twice():
    """
    Feature: Bernoulli function
    Description: Test cases for Bernoulli
    Expectation: No exception.
    """
    ms.set_seed(2)
    x = Tensor(np.random.randn(100, 100), ms.float32)
    op = ms.ops.bernoulli
    out1 = op(x).asnumpy()
    out2 = op(x).asnumpy()
    assert not np.allclose(out1, out2, rtol=0.0, atol=0.0, equal_nan=False)
