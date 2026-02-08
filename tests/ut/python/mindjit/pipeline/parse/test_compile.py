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
"""
@File  : test_compile.py
@Author:
@Date  : 2019-03-20
@Desc  : test mindspore compile method
"""
import logging
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Model, context
from mindspore.nn.optim import Momentum
from mindspore.ops.composite import add_flags
from tests.ut.python.ut_filter import non_graph_engine

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        out = self.flatten(x)
        return out


loss = nn.MSELoss()


# Test case 1 : test the new compiler interface
# _build_train_graph is deprecated
def test_build():
    """ test_build """
    Tensor(np.random.randint(0, 255, [1, 3, 224, 224]))
    Tensor(np.random.randint(0, 10, [1, 10]))
    net = Net()
    opt = Momentum(net.get_parameters(), learning_rate=0.1, momentum=0.9)
    Model(net, loss_fn=loss, optimizer=opt, metrics=None)


# Test case 2 : test the use different args to run graph
class Net2(nn.Cell):
    """ Net2 definition """

    def __init__(self):
        super(Net2, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x


@non_graph_engine
def test_different_args_run():
    """ test_different_args_run """
    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me1 = Tensor(np1)
    np2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me2 = Tensor(np2)

    net = Net2()
    net = add_flags(net, predit=True)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    me1 = model.predict(input_me1)
    me2 = model.predict(input_me2)
    out_me1 = me1.asnumpy()
    out_me2 = me2.asnumpy()
    print(np1)
    print(np2)
    print(out_me1)
    print(out_me2)
    assert not np.allclose(out_me1, out_me2, 0.01, 0.01)
