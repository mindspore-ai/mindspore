# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test_net_infer """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
import mindspore.ops.operations as op


def test_net_infer():
    """ test_net_infer """

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
            self.bn = nn.BatchNorm2d(64)
            self.fc = nn.Dense(64, 10)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()

        def construct(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.flatten(x)
            out = self.fc(x)
            return out

    Tensor(np.random.randint(0, 255, [1, 3, 224, 224]))
    Net()


def test_assign_in_while():
    context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def __init__(self, input_shape):
            super().__init__()
            self.assign = op.Assign()
            self.inputdata = Parameter(initializer(1, input_shape), name="global_step")

        def construct(self, x, y, z):
            out = z
            while x < y:
                inputdata = self.inputdata
                x = x + 1
                self.assign(inputdata, z)
                out = inputdata
            return out

    x = Tensor(np.array(1).astype(np.int32))
    y = Tensor(np.array(3).astype(np.int32))
    input_shape = (1024, 512)
    z = Tensor(np.random.randn(*input_shape).astype(np.float32))
    net = Net(input_shape)
    net(x, y, z)


def test_dup_context():
    """ different func_with_fv in net1 and net2 should produce 2 different FuncGraphAbstractClosure and
        Evaluator.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            def identity(f):
                return f

            def func_with_fv():
                return x

            def net1():
                local_func = identity(func_with_fv)
                out = local_func() + 20.0
                return out

            def net2():
                local_func = identity(func_with_fv)
                out = local_func() + 15.0
                return out

            return net1() + net2()

    Net()(Tensor(np.array(5.0).astype(np.float32)))


def test_maybe_poly_func():
    """ different func_with_fv in net1 and net2 may produce poly node. """
    context.set_context(mode=context.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y, z):
            def identity(f, inp):
                return f(inp)

            def func_with_fv(yy):
                return (x, yy)

            def make_call():
                out1 = identity(func_with_fv, y)
                out2 = identity(func_with_fv, z)
                return (out1, out2)

            return make_call()

    y_input = Tensor(np.array([1, 2]).astype(np.int32))
    z_input = Tensor(np.array([[2, 2], [3, 3]]).astype(np.int32))
    Net()(Tensor(np.array(1).astype(np.int32)), y_input, z_input)
