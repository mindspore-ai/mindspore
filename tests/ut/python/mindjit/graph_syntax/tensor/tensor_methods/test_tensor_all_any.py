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
""" test interface 'all' and 'any' of tensor """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


def test_all_and_any_of_tensor_in_graph():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            all_ = x.all()
            any_ = x.any()
            all_0 = x.all(None, True)
            any_0 = x.any(None, True)
            return all_, any_, all_0, any_0

    net = Net()
    x = Tensor(np.array([[True, False, False], [True, False, False]]))
    context.set_context(mode=context.GRAPH_MODE)
    net(x)


def test_all_and_any_of_tensor_in_pynative():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            all_ = x.all()
            any_ = x.any()
            all_0 = x.all(0, True)
            any_0 = x.any(0, True)
            return all_, any_, all_0, any_0

    net = Net()
    x = Tensor(np.array([[True, False, True], [True, False, False]]))
    context.set_context(mode=context.PYNATIVE_MODE)
    ret = net(x)
    assert ret[0].asnumpy() == np.array(False)
    assert ret[1].asnumpy() == np.array(True)
    assert ret[2].asnumpy().shape == np.array([[True, False, False]]).shape
    assert (ret[2].asnumpy() == np.array([[True, False, False]])).all()
    assert ret[3].shape == Tensor(np.array([[True, False, True]])).shape
    assert (ret[3] == Tensor(np.array([[True, False, True]]))).all()
