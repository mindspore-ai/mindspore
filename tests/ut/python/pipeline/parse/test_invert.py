# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test '~' """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class InvertNet(nn.Cell):
    def __init__(self):
        super(InvertNet, self).__init__()
        self.t = Tensor(np.array([True, False, True]))

    def construct(self, x):
        invert_t = ~self.t
        invert_x = ~x
        ret = (invert_t, invert_x)
        return ret


def test_invert_bool_tensor():
    net = InvertNet()
    input_x = Tensor(np.array([False, True, False]))

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = net(input_x)
    assert (ret[0].asnumpy() == np.array([False, True, False])).all()
    assert (ret[1].asnumpy() == np.array([True, False, True])).all()

    context.set_context(mode=context.GRAPH_MODE)
    net(input_x)


def test_invert_int_tensor():
    net = InvertNet()
    input_x = Tensor(np.array([1, 2, 3], np.int32))

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(TypeError) as err:
        net(input_x)
    assert "For 'LogicalNot or '~' operator', the type of `x` should be subclass of Tensor[Bool], " \
           "but got Tensor[Int32]" in str(err.value)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(TypeError) as err:
        net(input_x)
    assert "For 'LogicalNot or '~' operator', the type of `x` should be subclass of Tensor[Bool], " \
           "but got Tensor[Int32]" in str(err.value)
