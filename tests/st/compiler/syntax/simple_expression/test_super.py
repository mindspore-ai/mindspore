# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
""" test syntax for logic expression """

import pytest
import numpy as np

import mindspore.nn as nn
import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class ArgumentNum(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    def construct(self, x, y):
        super(ArgumentNum, 2, 3).aa()
        out = self.matmul(x, y)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_super_argument_num():
    """
    Feature: simple expression
    Description: test super
    Expectation: No exception
    """
    with pytest.raises(RuntimeError) as err:
        x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        net = ArgumentNum()
        ret = net(x, y)
        print(ret)
    assert "Arguments number of 'super()' should be 0 or 2, but got 3." in str(err)


class ArgumentNotSelf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    def construct(self, x, y):
        super(ArgumentNotSelf, 2).aa()
        out = self.matmul(x, y)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_super_argument_not_self():
    """
    Feature: simple expression
    Description: test super
    Expectation: No exception
    """
    with pytest.raises(RuntimeError) as err:
        x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        net = ArgumentNotSelf()
        ret = net(x, y)
        print(ret)
    assert "Unable to cast Python instance of type <class 'NoneType'> to C++ type 'std::string'" in str(err)


class ArgumentType(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    def construct(self, x, y):
        super(ArgumentType, self).aa()
        out = self.matmul(x, y)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_super_argument_type():
    """
    Feature: simple expression
    Description: test super
    Expectation: No exception
    """
    with pytest.raises(AttributeError) as err:
        x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        net = ArgumentType()
        ret = net(x, y)
        print(ret)
    assert "External object has no attribute aa" in str(err)
